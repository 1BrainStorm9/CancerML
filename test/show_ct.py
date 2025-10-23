import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = 'test/data'
ANNOT_PATH = os.path.join(DATA_DIR, 'annotations_rescaled.csv')


def load_dicom_series(folder):
    slices = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(folder, fname))
            slices.append(ds.pixel_array)
    return np.stack(slices, axis=0)


def show_ct(all_series, start_series=0):
    if not all_series:
        print("❌ Нет обработанных КТ")
        return

    ann = pd.read_csv(ANNOT_PATH)

    current_series_idx = start_series
    current_slice = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title(f"CT Viewer — {all_series[current_series_idx]}")

    def draw_slice():
        nonlocal current_series_idx, current_slice

        seriesuid = all_series[current_series_idx]
        volume_path = os.path.join(DATA_DIR, seriesuid)
        volume = load_dicom_series(volume_path)
        nodules = ann[ann['seriesuid'] == seriesuid]

        ax.clear()
        ax.imshow(volume[current_slice], cmap='gray')
        plt.title(f"{seriesuid} — Slice {current_slice + 1}/{len(volume)}")
        plt.axis('off')

        # Отображаем все узелки, которые находятся на этом срезе
        nodules_on_slice = nodules[np.abs(nodules['coordZ'] - current_slice) < 1]
        for idx, nodule in nodules_on_slice.iterrows():
            y, x = nodule['coordY'], nodule['coordX']
            circ = plt.Circle((x, y), 5, color='red', fill=False, linewidth=1.5)
            ax.add_patch(circ)

        if len(nodules) > 0:
            plt.text(5, 15, f"Nodules: {len(nodules_on_slice)}/{len(nodules)}", color='red')

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current_series_idx, current_slice

        seriesuid = all_series[current_series_idx]
        volume_path = os.path.join(DATA_DIR, seriesuid)
        volume = load_dicom_series(volume_path)

        if event.key == 'up':
            current_slice = (current_slice - 1) % len(volume)
        elif event.key == 'down':
            current_slice = (current_slice + 1) % len(volume)
        elif event.key == 'left':
            current_series_idx = (current_series_idx - 1) % len(all_series)
            current_slice = 0
            fig.canvas.manager.set_window_title(f"CT Viewer — {all_series[current_series_idx]}")
        elif event.key == 'right':
            current_series_idx = (current_series_idx + 1) % len(all_series)
            current_slice = 0
            fig.canvas.manager.set_window_title(f"CT Viewer — {all_series[current_series_idx]}")
        elif event.key == 'escape':
            plt.close(fig)
            return

        draw_slice()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw_slice()
    plt.show()


def main():
    if not os.path.exists(ANNOT_PATH):
        print("❌ Файл аннотаций не найден:", ANNOT_PATH)
        return

    all_series = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    if not all_series:
        print("❌ Нет обработанных КТ в:", DATA_DIR)
        return

    print(f"🔍 Найдено {len(all_series)} КТ-сканов")
    show_ct(all_series)


if __name__ == "__main__":
    main()
