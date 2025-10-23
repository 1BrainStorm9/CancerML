import os
import pandas as pd

# Путь к твоей папке с raw/mhd
DATA_DIR = "data/LUNA16/raw"

# Путь к файлу аннотаций nodules.csv из LUNA16
ANNOTATION_CSV = "data/LUNA16/annotations.csv"

# Загружаем аннотации
df = pd.read_csv(ANNOTATION_CSV)

# Фильтруем только узелки >= 3 мм
df = df[df["diameter_mm"] >= 3.0]

# Получаем уникальные seriesuid с узелками
uids_with_nodules = set(df["seriesuid"].unique())

# Получаем уникальные seriesuid локальных файлов (по .mhd)
local_files = os.listdir(DATA_DIR)
local_uids = set(f.replace(".mhd", "") for f in local_files if f.endswith(".mhd"))

# Считаем сколько узелков доступно локально
available_uids = uids_with_nodules.intersection(local_uids)
missing_uids = uids_with_nodules - local_uids

print(f"Всего КТ с узелками >=3мм в аннотации: {len(uids_with_nodules)}")
print(f"Доступные локально КТ с узелками: {len(available_uids)}")
print(f"Отсутствуют локально: {len(missing_uids)}")
