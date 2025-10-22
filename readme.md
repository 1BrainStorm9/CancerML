# LUNA16 Lung Nodule Segmentation with 3D U-Net

Production-ready проект для сегментации (локализации) узелков в КТ-снимках лёгких с использованием 3D U-Net на датасете LUNA16.

## 📋 Описание

Проект реализует полный пайплайн для обучения модели глубокого обучения на задачу voxel-wise сегментации узелков лёгких:

- **Предобработка**: загрузка .mhd/.raw файлов, ресэмплинг к spacing 1x1x1 мм, нормализация HU, изменение размера к 256×256×64
- **Аугментации**: вращения, отражения, шум, масштабирование интенсивности, аффинные трансформации
- **Модель**: 3D U-Net с batch normalization, PReLU активациями
- **Обучение**: Dice + BCE loss, AdamW оптимизатор, Cosine Annealing LR scheduler, mixed precision training
- **Валидация**: подробные метрики (Dice, Sensitivity, Precision, Recall)
- **Инференс**: предсказание масок, извлечение координат узелков, визуализация

## 🗂️ Структура проекта

```
luna16_segmentation/
├── README.md                  # Документация
├── requirements.txt           # Зависимости
├── utils.py                   # Вспомогательные функции
├── preprocess.py              # Предобработка данных
├── dataset.py                 # PyTorch Dataset
├── model.py                   # 3D U-Net модель
├── train.py                   # Скрипт обучения
├── validate.py                # Скрипт валидации
├── infer.py                   # Скрипт инференса
└── data/
    └── LUNA16/
        ├── raw/               # Исходные .mhd/.raw файлы
        ├── annotations.csv    # Аннотации узелков
        ├── seg-lungs-LUNA16/  # Маски лёгких
        └── processed/         # Обработанные данные
```

## 🔧 Установка

### 1. Создание виртуального окружения

```bash
# С помощью conda (рекомендуется)
conda create -n luna16 python=3.10
conda activate luna16

# Или с помощью venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Установка зависимостей

```bash
# Установка PyTorch (выберите версию для вашей CUDA)
# Для CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Для CPU:
pip install torch torchvision torchaudio

# Установка остальных зависимостей
pip install -r requirements.txt
```

## 📊 Подготовка данных

### 1. Загрузка датасета LUNA16

Скачайте следующие файлы с [LUNA16 официального сайта](https://luna16.grand-challenge.org/Download/):

- **Субсеты 0-9**: файлы .mhd/.raw с КТ-снимками
- **annotations.csv**: координаты и диаметры узелков
- **seg-lungs-LUNA16.zip**: маски лёгких

### 2. Организация данных

Разместите файлы следующим образом:

```
data/LUNA16/
├── raw/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.*.mhd
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.*.raw
│   └── ...
├── annotations.csv
└── seg-lungs-LUNA16/
    ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.*.mhd
    └── ...
```

### 3. Предобработка

Запустите скрипт предобработки:

```bash
python preprocess.py
```

Параметры предобработки (можно изменить в `preprocess.py`):
- **target_spacing**: (1.0, 1.0, 1.0) мм
- **target_size**: (64, 256, 256) вокселей (z, y, x)
- **HU range**: [-1000, 400]

Результат: обработанные файлы в `data/LUNA16/processed/`

**⚠️ Важно**: После ресэмплинга и изменения размера мировые координаты автоматически пересчитываются и сохраняются в метаданных.

## 🚀 Обучение модели

### Базовое обучение

```bash
python train.py
```

### Параметры обучения (в `train.py`):

```python
trainer = Trainer(
    data_dir='data/LUNA16/processed',
    output_dir='experiments/luna16_unet',
    num_epochs=100,              # Количество эпох
    batch_size=2,                # Размер батча (зависит от GPU памяти)
    learning_rate=1e-4,          # Learning rate
    weight_decay=1e-5,           # Weight decay для AdamW
    num_workers=4,               # Количество workers для DataLoader
    device='cuda',               # 'cuda' или 'cpu'
    mixed_precision=True,        # Mixed precision training
    checkpoint_interval=5        # Интервал сохранения чекпоинтов
)
```

### Мониторинг обучения

```bash
# Запуск TensorBoard
tensorboard --logdir experiments/luna16_unet/logs_*
```

Откройте браузер: http://localhost:6006

### Результаты обучения

Сохраняются в `experiments/luna16_unet/`:
- `checkpoints/best_model.pth` - лучшая модель
- `checkpoints/checkpoint_epoch_*.pth` - промежуточные чекпоинты
- `logs_*/` - логи для TensorBoard

## 📈 Валидация

После обучения запустите детальную валидацию:

```bash
python validate.py
```

Результаты:
- `validation_results.csv` - метрики для каждого пациента
- `metrics_distribution.png` - визуализация распределения метрик
- Статистика в консоли (mean ± std для всех метрик)

## 🔮 Инференс

Для получения предсказаний на новых данных:

```bash
python infer.py
```

Результаты в `experiments/luna16_unet/inference/`:
- `predicted_masks/` - предсказанные маски в формате .npy
- `visualizations/` - визуализация предсказаний (срезы с наложением)
- `*_nodules.npy` - координаты и характеристики обнаруженных узелков
- `inference_summary.npy` - сводная информация

### Формат выходных данных