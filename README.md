# mados-develop

Исследование методов машинного обучения в задачах семантической сегментации мультиспектральных спутниковых изображений (Sentinel-2) на наборе данных **MADOS** (Marine Debris and Oil Spill). Курсовой проект по дисциплине «Технологии и инструментальные средства проектирования интеллектуальных систем», БрГТУ, кафедра ИИТ.

Базовая архитектура: **MariNeXt** (энкодер MSCAN + декодер LightHamHead), 11 спектральных каналов, 15 классов, размер патча 240×240.

## Что сделано

- Подготовлен набор MADOS в формате NumPy (`--data_format numpy`).
- Добавлены адаптивные нормализации (ALN / ABN / AGN) и активация TReLU.
- Реализованы Dice loss, AdamW, cosine-расписание, `ReduceLROnPlateau`, EMA-модель.
- Проведена серия ablation-запусков (run1–run7) при фиксированных данных и `seed = 0`.
- Результаты сравнены по IoU и macro F1 (в том числе для EMA-модели), проанализированы признаки переобучения.

## Структура репозитория

```
mados-develop/
├── marinext/
│   ├── configs/            # конфигурации моделей (mados, aln, abn, agn и их комбинации)
│   ├── mmseg/              # модули сегментации (backbones, decode_heads, losses)
│   ├── train.py            # обучение
│   ├── evaluation.py       # оценка на val / test
│   ├── model_summary.py    # сводка по модели
│   └── trelu_prune_eval.py # оценка с прунингом TReLU
├── utils/                  # датасет, метрики, аугментации, спектральные индексы
├── figures/                # графики результатов экспериментов
├── train.sh                # пример запуска обучения
├── eval.sh                 # пример запуска оценки
├── model_summary.sh
├── trelu_prune_eval.sh
└── requirements.txt
```

## Установка

```bash
git clone https://github.com/toritoos/mados-develop.git
cd mados-develop
conda create -n mados python=3.8 -y
conda activate mados
pip install -r requirements.txt
```

Основные зависимости: `torch 1.11 (CUDA 11.3)`, `mmcv-full 1.6.0`, `timm 0.4.12`, `numpy`, `scikit-learn`, `tensorboard`.

## Данные

Набор MADOS: <https://doi.org/10.5281/zenodo.10664073>

Для формата NumPy ожидается структура:

```
data/MADOS/
├── train/   # *.npy
├── val/     # *.npy
└── test/    # *.npy
```

## Обучение

```bash
python -m marinext.train \
  --path ./data/MADOS/ \
  --data_format numpy \
  --config marinext.tiny.240x240.mados_aln_abn_agn.py \
  --epochs 120 --batch 4 --seed 0
```

Основные параметры:

| Параметр | Назначение |
|---|---|
| `--config` | конфигурация модели из `marinext/configs` |
| `--optimizer` | `adam` или `adamw` |
| `--scheduler` | `none` или `cosine` |
| `--dice_weight` | вес Dice loss (0 — только CE) |
| `--add_val` | добавить val в обучающую выборку |
| `--model_ema` | использовать EMA-модель |
| `--reduce_lr_on_plateau` | включить `ReduceLROnPlateau` |
| `--resume_from_epoch` | продолжить с эпохи |

Метрики и кривые обучения пишутся в TensorBoard / MLflow.

## Оценка

```bash
python -m marinext.evaluation \
  --path ./data/MADOS/ \
  --data_format numpy \
  --model_path <путь_к_чекпоинту> \
  --config marinext.tiny.240x240.mados_aln_abn_agn.py \
  --split test
```

## Эксперименты

| Запуск | Что менялось |
|---|---|
| run1 | конфигурация `mados` (TReLU, без adaptive norms), Adam, MultiStepLR |
| run2 | базовая линия с ALN/ABN/AGN |
| run3 | + Dice (0.5) с первой эпохи |
| run4 | AdamW + cosine |
| run5 | AdamW + cosine + Dice |
| run6 | двухфазная схема: старт с весов run4, дообучение с Dice |
| run7 | обучение на train + val (`add_val=True`) |

### Итоги на 120-й эпохе

| Запуск | IoU | IoU_ema | macro F1 | macro F1_ema |
|---|---|---|---|---|
| run1 | 0,620 | 0,621 | 0,752 | 0,749 |
| run2 | 0,677 | 0,670 | 0,796 | 0,788 |
| run3 | 0,611 | 0,637 | 0,738 | 0,763 |
| run4 | 0,672 | **0,706** | 0,790 | **0,815** |
| run5 | 0,602 | 0,646 | 0,727 | 0,766 |
| run6 | 0,642 | 0,680 | 0,764 | 0,798 |
| run7 | 0,700 | 0,696 | 0,802 | 0,801 |

### Выводы

- Лучший итог на 120-й эпохе у **run4** (AdamW + cosine): IoU_ema = 0,706, macro F1_ema = 0,815.
- Adaptive norms дали прирост относительно конфигурации без них (run1 → run2/run4).
- Dice loss в испытанных схемах качество не повысил.
- У части запусков val loss растёт при снижении train loss, что указывает на переобучение при длинном горизонте.
- Все результаты получены при одном значении seed; различия порядка 0,02–0,03 без повторных запусков не считаются строго подтверждёнными.

Графики лежат в папке `figures/`.

## Автор

Сугак В. А., группа ИИ-26, ФЭИС, БрГТУ. Руководитель: Андренко К. В.
