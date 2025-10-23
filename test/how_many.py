import os

OUTPUT_DIR = 'test/data'
POS_DIR = os.path.join(OUTPUT_DIR, 'positive')
NEG_DIR = os.path.join(OUTPUT_DIR, 'negative')
RAW_DIR = 'data/LUNA16/raw'

def count_scans(raw_dir, pos_dir, neg_dir):
    # Подсчёт исходных файлов .mhd
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.mhd')]
    print(f"Всего исходных файлов .mhd: {len(raw_files)}")

    # Подсчёт позитивных КТ (с узелками)
    if os.path.exists(pos_dir):
        pos_series = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d))]
        print(f"Позитивные КТ (с узелками): {len(pos_series)}")
    else:
        print("Позитивные КТ (с узелками): 0")

    # Подсчёт негативных КТ (без узелков)
    if os.path.exists(neg_dir):
        neg_series = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d))]
        print(f"Негативные КТ (без узелков): {len(neg_series)}")
    else:
        print("Негативные КТ (без узелков): 0")

    # Общее количество конвертированных КТ
    total_converted = len(pos_series) + len(neg_series)
    print(f"Всего конвертированных КТ: {total_converted}")


if __name__ == "__main__":
    count_scans(RAW_DIR, POS_DIR, NEG_DIR)
