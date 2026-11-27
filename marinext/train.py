import os
import ast
import sys
import json
import random
import logging
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import functional as F
from timm.utils import ModelEma

import mlflow

sys.path.append(up(os.path.abspath(__file__)))
from marinext_wrapper import MariNext

sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'utils'))
from dataset import MADOS, build_dataset, gen_weights, class_distr
from vscp import VSCP
from metrics import Evaluation
from assets import bool_flag, cosine_scheduler

root_path = up(up(os.path.abspath(__file__)))
time_now = str(time())
logging.basicConfig(filename=os.path.join(root_path, 'logs', 'log_marinext_' + time_now + '.log'), filemode='a',
                    level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*' * 10)

# MLflow writes into existing project-root/mlruns/
mlflow.set_tracking_uri(__import__('pathlib').Path(root_path, 'mlruns').as_uri())


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dice_loss(logits, target, num_classes, ignore_index=-1, eps=1e-6):
    """Soft multi-class Dice loss. Pixels with target == ignore_index are skipped.
    Classes that are absent from the batch are not averaged in."""
    probs = torch.softmax(logits, dim=1)
    valid = (target != ignore_index)
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    vf = valid.unsqueeze(1).float()
    probs = probs * vf
    onehot = onehot * vf
    inter = (probs * onehot).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    present = onehot.sum((0, 2, 3)) > 0
    if present.sum() == 0:
        return logits.sum() * 0.0
    return 1 - dice[present].mean()


###############################################################
# Training                                                    #
###############################################################

def main(options):
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior
    seed_all(options['seed'])
    g = torch.Generator()
    g.manual_seed(options['seed'])

    # Tensorboard
    writer = SummaryWriter(os.path.join(root_path, 'logs', options['tensorboard'] + '_' + time_now))

    splits_path = os.path.join(options['path'], 'splits')

    # Construct Data loader
    dataset_train = build_dataset(options['path'], 'train', options['data_format'])
    if options['add_val']:
        # train on train + val; the test split stays untouched and is used for evaluation only
        dataset_extra = build_dataset(options['path'], 'val', options['data_format'])
        dataset_extra.mode = 'train'  # otherwise val patches would be used without augmentation
        dataset_train = ConcatDataset([dataset_train, dataset_extra])
    # dataset_val = MADOS(options['path'], splits_path, 'val')
    dataset_val = build_dataset(options['path'], 'test', options['data_format'])

    train_loader = DataLoader(dataset_train,
                              batch_size=options['batch'],
                              shuffle=True,
                              num_workers=options['num_workers'],
                              pin_memory=options['pin_memory'],
                              prefetch_factor=options['prefetch_factor'],
                              persistent_workers=options['persistent_workers'],
                              worker_init_fn=seed_worker,
                              generator=g,
                              drop_last=True)

    val_loader = DataLoader(dataset_val,
                            batch_size=options['batch'],
                            shuffle=False,
                            num_workers=options['num_workers'],
                            pin_memory=options['pin_memory'],
                            prefetch_factor=options['prefetch_factor'],
                            persistent_workers=options['persistent_workers'],
                            worker_init_fn=seed_worker,
                            generator=g)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = MariNext(options['input_channels'], options['output_channels'], options['config'])

    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    if options['resume_from_epoch'] >= 1:

        resume_model_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_model_dir, 'model.pth')
        logging.info('Loading model files from folder: %s' % model_file)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_ema = None
    if options['model_ema']:
        model_ema = ModelEma(
            model,
            decay=options['model_ema_decay'],
            device=device,
            resume='')

        ema_decay_schedule = cosine_scheduler(options['model_ema_decay'], 0.999,
                                              options['epochs'], len(train_loader))

        print("Using EMA with decay = %.8f" % options['model_ema_decay'])

    # Weighted Cross Entropy Loss & optimizer
    weight = gen_weights(class_distr, c=options['weight_param'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight.to(device),
                                          label_smoothing=options['label_smoothing'])

    if options['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=options['decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])

    # Learning Rate scheduler
    if options['scheduler'] == 'cosine':
        warmup_epochs = 5
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(options['epochs'] - warmup_epochs, 1), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])
    elif options['reduce_lr_on_plateau'] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=options['factor'], patience=options['patience'], verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options['lr_steps'], gamma=0.1, verbose=True)

    # Start training
    start = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    # Write model-graph to Tensorboard
    if options['mode'] == 'train':

        mlflow.set_experiment(options.get('experiment_name', 'marinext'))
        with mlflow.start_run(run_name=options.get('run_name', options['tensorboard'])):
            mlflow.log_params({k: str(v) for k, v in options.items()})

            ###############################################################
            # Start Training                                              #
            ###############################################################
            model.train()

            for epoch in range(start, epochs + 1):

                training_loss = []
                training_batches = 0

                i_board = 0
                for it, (image, target) in enumerate(tqdm(train_loader, desc="training")):

                    it = len(train_loader) * (epoch - 1) + it  # global training iteration

                    if options['vscp']:
                        image_augmented, target_augmented = VSCP(image.cpu().numpy(), target.cpu().numpy())

                        image = torch.cat([image, torch.tensor(image_augmented).to(image.device)])
                        target = torch.cat([target, torch.tensor(target_augmented).to(target.device)])

                    image = image.to(device)
                    target = target.long().to(device)

                    optimizer.zero_grad()

                    logits = F.upsample(input=model(image),
                                        size=image.size()[2:4], mode='bilinear')

                    # ce is logged (comparable with baseline runs), loss is what is optimised
                    ce = criterion(logits, target)
                    loss = ce
                    if options['dice_weight'] > 0:
                        loss = ce + options['dice_weight'] * dice_loss(logits, target, options['output_channels'])

                    loss.backward()

                    training_batches += target.shape[0]

                    training_loss.append((ce.data * target.shape[0]).tolist())

                    if options['clip_grad'] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), options['clip_grad'])

                    optimizer.step()

                    if model_ema is not None:
                        model_ema.decay = ema_decay_schedule[it]
                        model_ema.update(model)

                    # Write running loss
                    writer.add_scalar('training loss', ce, (epoch - 1) * len(train_loader) + i_board)
                    i_board += 1

                train_loss_epoch = sum(training_loss) / training_batches
                logging.info("Training loss was: " + str(train_loss_epoch))
                mlflow.log_metric('train_loss', train_loss_epoch, step=epoch)

                logging.info("Saving models")
                model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

                ###############################################################
                # Start Evaluation                                            #
                ###############################################################
                if epoch % eval_every == 0 or epoch == 1:
                    model.eval()

                    val_loss = []
                    val_batches = 0
                    y_true_val = []
                    y_predicted_val = []

                    seed_all(0)

                    with torch.no_grad():
                        for (image, target) in tqdm(val_loader, desc="validating"):
                            image = image.to(device)
                            target = target.to(device)

                            logits = model(image)
                            logits = F.upsample(input=logits, size=(
                                target.shape[-2], target.shape[-1]), mode='bilinear')

                            loss = criterion(logits, target)

                            # Accuracy metrics only on annotated pixels
                            logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                            logits = logits.reshape((-1, options['output_channels']))
                            target = target.reshape(-1)
                            mask = target != -1
                            logits = logits[mask]
                            target = target[mask]

                            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                            target = target.cpu().numpy()

                            val_batches += target.shape[0]
                            val_loss.append((loss.data * target.shape[0]).tolist())
                            y_predicted_val += probs.argmax(1).tolist()
                            y_true_val += target.tolist()

                        y_predicted_val = np.asarray(y_predicted_val)
                        y_true_val = np.asarray(y_true_val)

                        ####################################################################
                        # Save Scores to the .log file and visualize also with tensorboard #
                        ####################################################################

                        acc_val = Evaluation(y_predicted_val, y_true_val)

                    logging.info("\n")
                    logging.info("Evaluating model..")
                    logging.info("Val loss was: " + str(sum(val_loss) / val_batches))
                    logging.info("RESULTS AFTER EPOCH " + str(epoch) + ": \n")
                    logging.info("Evaluation: " + str(acc_val))

                    writer.add_scalars('Loss per epoch', {'Val loss': sum(val_loss) / val_batches,
                                                          'Train loss': sum(training_loss) / training_batches},
                                       epoch)

                    writer.add_scalar('Precision/val macroPrec', acc_val["macroPrec"], epoch)
                    writer.add_scalar('Precision/val microPrec', acc_val["microPrec"], epoch)
                    writer.add_scalar('Precision/val weightPrec', acc_val["weightPrec"], epoch)
                    writer.add_scalar('Recall/val macroRec', acc_val["macroRec"], epoch)
                    writer.add_scalar('Recall/val microRec', acc_val["microRec"], epoch)
                    writer.add_scalar('Recall/val weightRec', acc_val["weightRec"], epoch)
                    writer.add_scalar('F1/val macroF1', acc_val["macroF1"], epoch)
                    writer.add_scalar('F1/val microF1', acc_val["microF1"], epoch)
                    writer.add_scalar('F1/val weightF1', acc_val["weightF1"], epoch)
                    writer.add_scalar('IoU/val MacroIoU', acc_val["IoU"], epoch)

                    val_loss_epoch = sum(val_loss) / val_batches
                    mlflow.log_metric('val_loss', val_loss_epoch, step=epoch)
                    mlflow.log_metrics({
                        'macroPrec': acc_val['macroPrec'],
                        'microPrec': acc_val['microPrec'],
                        'weightPrec': acc_val['weightPrec'],
                        'macroRec': acc_val['macroRec'],
                        'microRec': acc_val['microRec'],
                        'weightRec': acc_val['weightRec'],
                        'macroF1': acc_val['macroF1'],
                        'microF1': acc_val['microF1'],
                        'weightF1': acc_val['weightF1'],
                        'IoU': acc_val['IoU'],
                    }, step=epoch)

                    if options['scheduler'] != 'cosine' and options['reduce_lr_on_plateau'] == 1:
                        scheduler.step(val_loss_epoch)
                    else:
                        scheduler.step()

                    model.train()

                ###############################################################
                # Start EMA Evaluation                                            #
                ###############################################################

                if (epoch % eval_every == 0 or epoch == 1) and options['model_ema'] and options['model_ema_eval']:

                    logging.info("Saving models")
                    model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model_ema.ema.state_dict(), os.path.join(model_dir, 'model_ema.pth'))

                    val_loss_ema = []
                    val_batches_ema = 0
                    y_true_val_ema = []
                    y_predicted_val_ema = []

                    seed_all(0)

                    with torch.no_grad():
                        for (image, target) in tqdm(val_loader, desc="validating"):
                            image = image.to(device)
                            target = target.to(device)

                            logits = model_ema.ema(image)
                            logits = F.upsample(input=logits, size=(
                                target.shape[-2], target.shape[-1]), mode='bilinear')

                            loss = criterion(logits, target)

                            # Accuracy metrics only on annotated pixels
                            logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                            logits = logits.reshape((-1, options['output_channels']))
                            target = target.reshape(-1)
                            mask = target != -1
                            logits = logits[mask]
                            target = target[mask]

                            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                            target = target.cpu().numpy()

                            val_batches_ema += target.shape[0]
                            val_loss_ema.append((loss.data * target.shape[0]).tolist())
                            y_predicted_val_ema += probs.argmax(1).tolist()
                            y_true_val_ema += target.tolist()

                        y_predicted_val_ema = np.asarray(y_predicted_val_ema)
                        y_true_val_ema = np.asarray(y_true_val_ema)

                        ####################################################################
                        # Save Scores to the .log file and visualize also with tensorboard #
                        ####################################################################

                        acc_val_ema = Evaluation(y_predicted_val_ema, y_true_val_ema)

                    logging.info("\n")
                    logging.info("Evaluating EMA model..")
                    logging.info("val loss was: " + str(sum(val_loss_ema) / val_batches_ema))
                    logging.info("RESULTS AFTER EPOCH " + str(epoch) + ": \n")
                    logging.info("Evaluation: " + str(acc_val_ema))

                    writer.add_scalars('Loss per epoch (EMA)', {'val loss': sum(val_loss_ema) / val_batches_ema}, epoch)
                    writer.add_scalar('Precision/val macroPrec (EMA)', acc_val_ema["macroPrec"], epoch)
                    writer.add_scalar('Precision/val microPrec (EMA)', acc_val_ema["microPrec"], epoch)
                    writer.add_scalar('Precision/val weightPrec (EMA)', acc_val_ema["weightPrec"], epoch)
                    writer.add_scalar('Recall/val macroRec (EMA)', acc_val_ema["macroRec"], epoch)
                    writer.add_scalar('Recall/val microRec (EMA)', acc_val_ema["microRec"], epoch)
                    writer.add_scalar('Recall/val weightRec (EMA)', acc_val_ema["weightRec"], epoch)
                    writer.add_scalar('F1/val macroF1 (EMA)', acc_val_ema["macroF1"], epoch)
                    writer.add_scalar('F1/val microF1 (EMA)', acc_val_ema["microF1"], epoch)
                    writer.add_scalar('F1/val weightF1 (EMA)', acc_val_ema["weightF1"], epoch)
                    writer.add_scalar('IoU/val MacroIoU (EMA)', acc_val_ema["IoU"], epoch)

                    val_loss_ema_epoch = sum(val_loss_ema) / val_batches_ema
                    mlflow.log_metric('val_loss_ema', val_loss_ema_epoch, step=epoch)
                    mlflow.log_metrics({
                        'macroPrec_ema': acc_val_ema['macroPrec'],
                        'microPrec_ema': acc_val_ema['microPrec'],
                        'weightPrec_ema': acc_val_ema['weightPrec'],
                        'macroRec_ema': acc_val_ema['macroRec'],
                        'microRec_ema': acc_val_ema['microRec'],
                        'weightRec_ema': acc_val_ema['weightRec'],
                        'macroF1_ema': acc_val_ema['macroF1'],
                        'microF1_ema': acc_val_ema['microF1'],
                        'weightF1_ema': acc_val_ema['weightF1'],
                        'IoU_ema': acc_val_ema['IoU'],
                    }, step=epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', help='Path of the images')
    parser.add_argument('--data_format', default='tif', choices=['tif', 'numpy'],
                        help="tif = original MADOS GeoTIFFs, numpy = <path>/{train,val,test}/*.npy")
    parser.add_argument("--config", default='marinext.tiny.240x240.mados.py', type=str)
    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=120, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=8, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    parser.add_argument('--seed', default=0, type=int, help='Random seed (0 = original behaviour)')
    parser.add_argument('--add_val', type=bool_flag, default=False,
                        help='also train on the val split (test stays untouched); False = original behaviour')

    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=15, type=int, help='Number of output classes')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    # Optimization
    parser.add_argument('--vscp', type=bool_flag, default=True)
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing')
    parser.add_argument('--clip_grad', default=None, type=float, help='Gradient Cliping')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=1e-4, type=float, help='learning rate decay')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw'],
                        help='adam = original behaviour')
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine'],
                        help='none = original behaviour (MultiStepLR or ReduceLROnPlateau)')
    parser.add_argument('--dice_weight', default=0.0, type=float,
                        help='weight of the Dice loss added to cross-entropy (0 = original behaviour)')
    parser.add_argument('--reduce_lr_on_plateau', default=0, type=int,
                        help='reduce learning rate when no increase (0 or 1)')
    parser.add_argument('--factor', default=0.1, type=float, help='reduce learning rate factor for lr')
    parser.add_argument('--patience', default=10, type=int, help='reduce learning rate patience from 1 to epoch_count')
    parser.add_argument('--lr_steps', default='[30,50,70]', type=str,
                        help='Specify the steps that the lr will be reduced')

    # Evaluation/Checkpointing
    parser.add_argument('--checkpoint_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models'),
                        help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    # EMA related parameters
    parser.add_argument('--model_ema', type=bool_flag, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_eval', type=bool_flag, default=True, help='Using ema to eval during training.')

    # misc
    parser.add_argument('--num_workers', default=6, type=int,
                        help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=True, type=bool_flag, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Number of sample loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool_flag,
                        help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--tensorboard', default='tsboard_segm', type=str, help='Name for tensorboard run')
    parser.add_argument('--experiment_name', default='marinext', type=str, help='MLflow experiment name')
    parser.add_argument('--run_name', default=None, type=str, help='MLflow run name (default = tensorboard name)')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict

    # lr_steps list or single float
    lr_steps = ast.literal_eval(options['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise ValueError("lr_steps must be list or number")

    options['lr_steps'] = lr_steps
    if options.get('run_name') is None:
        options['run_name'] = options['tensorboard']

    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent=2))

    main(options)