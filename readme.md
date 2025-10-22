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

Каждый обнаруженный узелок содержит:
```python
{
    'center_world': [x, y, z],        # Мировые координаты центра (мм)
    'center_voxel': [x, y, z],        # Voxel координаты
    'diameter_mm': 8.5,               # Диаметр узелка (мм)
    'volume_voxels': 245,             # Объём в вокселях
    'bbox': (z_min, z_max, y_min, y_max, x_min, x_max)  # Bounding box
}
```

## 🔧 Настройка гиперпараметров

### Архитектура модели (`model.py`)

```python
model = LungNodule3DUNet(
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),  # Каналы на каждом уровне
    strides=(2, 2, 2, 2),              # Downsampling strides
    num_res_units=2,                    # Residual units
    dropout=0.1                         # Dropout probability
)
```

### Loss функция

```python
criterion = DiceBCELoss(
    dice_weight=0.5,   # Вес Dice loss
    bce_weight=0.5,    # Вес BCE loss
    smooth=1e-6        # Сглаживание
)
```

### Аугментации (`dataset.py`)

- **RandRotate90d**: вращение на 90° (prob=0.5)
- **RandFlipd**: отражение по осям (prob=0.5)
- **RandGaussianNoised**: гауссов шум (prob=0.3, std=0.05)
- **RandScaleIntensityd**: масштабирование интенсивности (prob=0.3, factors=0.1)
- **RandShiftIntensityd**: сдвиг интенсивности (prob=0.3, offsets=0.1)
- **RandAffined**: аффинные трансформации (prob=0.3)

## 📊 Метрики

Проект вычисляет следующие метрики:

1. **Dice Coefficient**: 
   ```
   Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
   ```

2. **Sensitivity (Recall, TPR)**:
   ```
   Sensitivity = TP / (TP + FN)
   ```

3. **Precision (PPV)**:
   ```
   Precision = TP / (TP + FP)
   ```

4. **Recall** (дублирует Sensitivity для полноты)

## ⚙️ Системные требования

### Минимальные
- **GPU**: NVIDIA GPU с 8+ GB VRAM (для batch_size=2)
- **RAM**: 16 GB
- **Диск**: 50 GB свободного места
- **CUDA**: 11.8 или выше

### Рекомендуемые
- **GPU**: NVIDIA RTX 3090 / A6000 (24 GB VRAM)
- **RAM**: 32 GB
- **Диск**: 100 GB SSD
- **CUDA**: 12.1

### Настройка для разных GPU

Для GPU с меньшей памятью:
```python
# В train.py
batch_size=1                    # Уменьшите batch size
channels=(16, 32, 64, 128, 256) # Уменьшите количество каналов

# В preprocess.py
target_size=(48, 192, 192)      # Уменьшите размер изображений
```

## 🐛 Устранение неполадок

### Ошибка: CUDA out of memory

**Решение**:
```python
# Уменьшите batch_size
batch_size = 1

# Или уменьшите размер изображения
target_size = (48, 192, 192)

# Или отключите mixed precision
mixed_precision = False
```

### Ошибка: FileNotFoundError при загрузке данных

**Решение**:
- Проверьте структуру директорий
- Убедитесь, что запущена предобработка (`python preprocess.py`)
- Проверьте наличие файла `processed_files.txt`

### Низкое качество предсказаний

**Решения**:
1. Увеличьте количество эпох обучения
2. Попробуйте другой learning rate (1e-3 или 5e-5)
3. Увеличьте вес Dice loss: `dice_weight=0.7, bce_weight=0.3`
4. Добавьте больше аугментаций
5. Проверьте баланс классов (узелки vs фон)

### SimpleITK не может прочитать файлы

**Решение**:
```bash
# Переустановите SimpleITK
pip uninstall SimpleITK
pip install SimpleITK>=2.2.0
```

## 📝 Примеры использования

### Быстрый старт (полный пайплайн)

```bash
# 1. Предобработка
python preprocess.py

# 2. Обучение (займёт несколько часов)
python train.py

# 3. Валидация
python validate.py

# 4. Инференс
python infer.py
```

### Обучение с кастомными параметрами

```python
# custom_train.py
from train import Trainer

trainer = Trainer(
    data_dir='data/LUNA16/processed',
    output_dir='experiments/custom_run',
    num_epochs=150,
    batch_size=4,
    learning_rate=5e-5,
    device='cuda',
    mixed_precision=True
)

trainer.train()
```

### Инференс на одном файле

```python
# single_inference.py
from infer import NoduleInferencer
import numpy as np

inferencer = NoduleInferencer(
    checkpoint_path='experiments/luna16_unet/checkpoints/best_model.pth',
    output_dir='results',
    device='cuda',
    threshold=0.5
)

# Загрузка данных
image = np.load('data/LUNA16/processed/images/sample.npy')

# Предсказание
mask_pred = inferencer.predict(image)

# Извлечение узелков
metadata = np.load('data/LUNA16/processed/sample_metadata.npy', allow_pickle=True).item()
nodules = inferencer.extract_nodule_candidates(
    mask_pred, 
    metadata['origin'], 
    metadata['spacing']
)

print(f"Detected {len(nodules)} nodules")
for i, nodule in enumerate(nodules):
    print(f"Nodule {i+1}: center={nodule['center_world']}, diameter={nodule['diameter_mm']:.1f}mm")
```

## 📚 Дополнительные ресурсы

### Датасет LUNA16
- [Официальный сайт](https://luna16.grand-challenge.org/)
- [Документация](https://luna16.grand-challenge.org/Data/)
- [Статья](https://arxiv.org/abs/1612.08012)

### MONAI
- [Документация](https://docs.monai.io/)
- [Tutorials](https://github.com/Project-MONAI/tutorials)
- [Model Zoo](https://github.com/Project-MONAI/model-zoo)

### 3D U-Net
- [Оригинальная статья](https://arxiv.org/abs/1606.06650)
- [Имплементация MONAI](https://docs.monai.io/en/stable/networks.html#unet)

## 🤝 Вклад в проект

Проект открыт для улучшений:

1. **Архитектура**: попробуйте nnU-Net, V-Net, Attention U-Net
2. **Loss функции**: Focal Loss, Tversky Loss, Boundary Loss
3. **Постобработка**: CRF, морфологические операции
4. **Ансамбли**: объединение нескольких моделей
5. **Метрики**: FROC анализ, Free-Response ROC

## 📄 Лицензия

Проект распространяется под лицензией MIT. Датасет LUNA16 имеет собственную лицензию - см. официальный сайт.

## ✉️ Контакты

При возникновении вопросов или проблем создавайте Issues в репозитории.

## 🎯 Ожидаемые результаты

После обучения на полном датасете LUNA16 (888 КТ-сканов) в течение 100 эпох (~8-12 часов на RTX 3090):

- **Dice Score**: 0.70-0.85
- **Sensitivity**: 0.75-0.90
- **Precision**: 0.65-0.80

Результаты зависят от:
- Качества предобработки
- Гиперпараметров
- Количества эпох обучения
- Баланса аугментаций

## 🔄 Версионирование

- **v1.0.0** (текущая) - базовая реализация с 3D U-Net
- Планируется: multi-scale inference, test-time augmentation, FROC метрики

---

**Удачи в обучении модели! 🚀**