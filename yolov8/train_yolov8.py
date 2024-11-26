from ultralytics import YOLO
import torch
import os
import shutil
from datetime import datetime
import yaml
import logging
import json
from pathlib import Path


class BallDetectionTrainer:
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        self.setup_paths()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Load training configuration"""
        default_config = {
            'model_type': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
            'image_size': 1080,
            'batch_size': 16,
            'epochs': 100,
            'patience': 15,
            'train_params': {
                'conf': 0.1,  # Initial confidence threshold
                'iou': 0.3,  # IoU threshold
                'max_det': 1,  # Maximum detections per image
                'rect': True,  # Rectangular training
                'mosaic': 0.3,  # Mosaic augmentation
                'scale': 0.3,  # Scale augmentation
                'degrees': 10,  # Rotation augmentation
                'warmup_epochs': 3,  # Warmup epochs
                'close_mosaic': 10,  # Disable mosaic in final epochs
                'optimizer': 'auto',  # Optimizer
                'seed': 42,  # Random seed
                'deterministic': True,  # Deterministic mode
                'single_cls': True,  # Single class training
            }
        }

        config_path = Path('config/training_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)

        return default_config

    def setup_paths(self):
        """Setup directory structure"""
        self.paths = {
            'data': Path('../dataset/yolo_dataset'),
            'models': Path('models'),
            'runs': Path('runs')
        }

        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def validate_dataset(self):
        """Validate dataset structure and contents"""
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']

        for dir_path in required_dirs:
            full_path = self.paths['data'] / dir_path
            if not full_path.exists():
                raise ValueError(f"Missing required directory: {full_path}")

        # Check if yaml exists
        yaml_path = self.paths['data'] / 'dataset.yaml'
        if not yaml_path.exists():
            raise ValueError(f"dataset.yaml not found in {self.paths['data']}")

        # Verify image-label pairs
        train_images = list((self.paths['data'] / 'train/images').glob('*.jpg'))
        train_labels = list((self.paths['data'] / 'train/labels').glob('*.txt'))

        self.logger.info(f"Found {len(train_images)} training images")
        self.logger.info(f"Found {len(train_labels)} training labels")

        if len(train_images) == 0 or len(train_labels) == 0:
            raise ValueError("No training data found!")

    def train(self):
        """Train the model with validation and checkpointing"""
        try:
            self.logger.info("Starting training pipeline...")
            self.validate_dataset()

            self.logger.info(f"Loading model: {self.config['model_type']}")
            model = YOLO(self.config['model_type'])

            # Training
            self.logger.info("Starting training...")
            results = model.train(
                data=str(self.paths['data'] / 'dataset.yaml'),
                epochs=self.config['epochs'],
                imgsz=self.config['image_size'],
                batch=self.config['batch_size'],
                device=0 if torch.cuda.is_available() else 'cpu',
                patience=self.config['patience'],
                project=str(self.paths['runs']),
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exist_ok=False,
                pretrained=True,
                **self.config['train_params']
            )

            # Save training results
            results_path = self.paths['runs'] / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)

            self.logger.info("Training completed successfully")

            # Validate final model
            self.logger.info("Running validation on test set...")
            val_results = model.val()

            self.logger.info(f"Validation results: mAP@0.5 = {val_results.results_dict['metrics/mAP50(B)']:.3f}")

            return True

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


if __name__ == "__main__":
    trainer = BallDetectionTrainer()
    trainer.train()