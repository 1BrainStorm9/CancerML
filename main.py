"""
Главный скрипт для запуска полного пайплайна LUNA16 сегментации узелков.
Позволяет запустить предобработку, обучение, валидацию и инференс одной командой.
"""

import argparse
import sys
from pathlib import Path

from config import ProjectConfig, default_config, get_small_model_config, get_large_model_config, get_fast_training_config
from preprocess import LUNA16Preprocessor
from train import Trainer
from validate import Validator
from infer import NoduleInferencer


def run_preprocessing(config: ProjectConfig):
    """Запуск предобработки данных."""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70 + "\n")
    
    preprocessor = LUNA16Preprocessor(
        raw_data_dir=config.data.raw_data_dir,
        annotations_path=config.data.annotations_path,
        lung_masks_dir=config.data.lung_masks_dir,
        output_dir=config.data.processed_dir,
        target_spacing=config.data.target_spacing,
        target_size=config.data.target_size,
        hu_min=config.data.hu_min,
        hu_max=config.data.hu_max
    )
    
    preprocessor.process_all()
    
    print("\n✓ Preprocessing completed successfully!\n")


def run_training(config: ProjectConfig):
    """Запуск обучения модели."""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70 + "\n")
    
    trainer = Trainer(
        data_dir=config.data.processed_dir,
        output_dir=config.output_dir,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_workers=config.training.num_workers,
        device=config.training.device,
        mixed_precision=config.training.mixed_precision,
        checkpoint_interval=config.training.checkpoint_interval
    )
    
    trainer.train()
    
    print("\n✓ Training completed successfully!\n")


def run_validation(config: ProjectConfig):
    """Запуск валидации модели."""
    print("\n" + "="*70)
    print("STEP 3: MODEL VALIDATION")
    print("="*70 + "\n")
    
    checkpoint_path = Path(config.output_dir) / 'checkpoints' / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run training first!")
        return
    
    validator = Validator(
        data_dir=config.data.processed_dir,
        checkpoint_path=str(checkpoint_path),
        output_dir=str(Path(config.output_dir) / 'validation'),
        device=config.inference.device,
        threshold=config.inference.threshold
    )
    
    validator.run_validation()
    
    print("\n✓ Validation completed successfully!\n")


def run_inference(config: ProjectConfig):
    """Запуск инференса."""
    print("\n" + "="*70)
    print("STEP 4: INFERENCE")
    print("="*70 + "\n")
    
    checkpoint_path = Path(config.output_dir) / 'checkpoints' / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run training first!")
        return
    
    inferencer = NoduleInferencer(
        checkpoint_path=str(checkpoint_path),
        output_dir=config.inference.output_dir,
        device=config.inference.device,
        threshold=config.inference.threshold,
        min_nodule_size=config.inference.min_nodule_size
    )
    
    inferencer.infer_dataset(data_dir=config.data.processed_dir)
    
    print("\n✓ Inference completed successfully!\n")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description='LUNA16 Lung Nodule Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Запуск полного пайплайна
  python main.py --all
  
  # Только предобработка
  python main.py --preprocess
  
  # Только обучение
  python main.py --train
  
  # Обучение + валидация
  python main.py --train --validate
  
  # Использование кастомной конфигурации
  python main.py --all --config small
  
  # Быстрое тестирование (малое количество эпох)
  python main.py --train --config fast
        """
    )
    
    # Этапы пайплайна
    parser.add_argument('--all', action='store_true',
                       help='Запустить весь пайплайн (preprocess + train + validate + infer)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Запустить предобработку данных')
    parser.add_argument('--train', action='store_true',
                       help='Запустить обучение модели')
    parser.add_argument('--validate', action='store_true',
                       help='Запустить валидацию модели')
    parser.add_argument('--infer', action='store_true',
                       help='Запустить инференс')
    
    # Конфигурация
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'small', 'large', 'fast'],
                       help='Выбор конфигурации модели')
    
    # Дополнительные параметры
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Устройство для обучения (переопределяет конфиг)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Размер батча (переопределяет конфиг)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох (переопределяет конфиг)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Директория для результатов (переопределяет конфиг)')
    
    args = parser.parse_args()
    
    # Если не указаны этапы, показываем help
    if not any([args.all, args.preprocess, args.train, args.validate, args.infer]):
        parser.print_help()
        sys.exit(0)
    
    # Загрузка конфигурации
    if args.config == 'default':
        config = default_config
    elif args.config == 'small':
        config = get_small_model_config()
    elif args.config == 'large':
        config = get_large_model_config()
    elif args.config == 'fast':
        config = get_fast_training_config()
    
    # Переопределение параметров из командной строки
    if args.device is not None:
        config.training.device = args.device
        config.inference.device = args.device
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        config.inference.output_dir = str(Path(args.output_dir) / 'inference')
    
    # Печать конфигурации
    config.print_config()
    
    # Запуск этапов пайплайна
    try:
        if args.all or args.preprocess:
            run_preprocessing(config)
        
        if args.all or args.train:
            run_training(config)
        
        if args.all or args.validate:
            run_validation(config)
        
        if args.all or args.infer:
            run_inference(config)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
        # Печать итоговой информации
        print("Results saved to:")
        print(f"  - Output directory: {config.output_dir}")
        
        if args.all or args.train:
            print(f"  - Model checkpoint: {config.output_dir}/checkpoints/best_model.pth")
            print(f"  - Training logs: {config.output_dir}/logs_*/")
        
        if args.all or args.validate:
            validation_dir = Path(config.output_dir) / 'validation'
            print(f"  - Validation results: {validation_dir}/validation_results.csv")
            print(f"  - Metrics plot: {validation_dir}/metrics_distribution.png")
        
        if args.all or args.infer:
            print(f"  - Inference results: {config.inference.output_dir}/")
            print(f"  - Predicted masks: {config.inference.output_dir}/predicted_masks/")
            print(f"  - Visualizations: {config.inference.output_dir}/visualizations/")
        
        print("\nNext steps:")
        if args.preprocess and not (args.train or args.all):
            print("  - Run training: python main.py --train")
        elif args.train and not (args.validate or args.all):
            print("  - Run validation: python main.py --validate")
            print("  - Run inference: python main.py --infer")
        
        print("\nFor monitoring training progress:")
        print(f"  tensorboard --logdir {config.output_dir}/logs_*")
        print()
        
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()