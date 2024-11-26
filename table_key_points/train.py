# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from model import KeypointModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


class PingPongTableDataset(Dataset):
    """Dataset for ping pong table keypoint detection."""

    def __init__(
            self,
            images_dir: Path,
            annotations_dir: Path,
            image_size: Tuple[int, int] = (224, 224),
            max_samples: Optional[int] = None
    ):
        self.image_size = image_size
        self.keypoint_names = [
            "LEFT_TOP", "RIGHT_TOP", "LEFT_BOTTOM",
            "RIGHT_BOTTOM"
        ]

        # Validate directories
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        if not self.images_dir.exists():
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory does not exist: {self.annotations_dir}")

        # Find valid image-annotation pairs
        self.image_paths, self.annotation_paths = self._find_valid_pairs(max_samples)
        logging.info(f"Found {len(self)} valid image-annotation pairs")

    def _find_valid_pairs(self, max_samples: Optional[int]) -> Tuple[List[Path], List[Path]]:
        """Find matching image and annotation files."""
        image_paths = []
        annotation_paths = []

        for anno_path in sorted(self.annotations_dir.glob("*.json")):
            img_path = self.images_dir / f"{anno_path.stem}.jpg"
            if img_path.exists():
                image_paths.append(img_path)
                annotation_paths.append(anno_path)

        if max_samples:
            image_paths = image_paths[:max_samples]
            annotation_paths = annotation_paths[:max_samples]

        if not image_paths:
            raise ValueError("No valid image-annotation pairs found!")

        return image_paths, annotation_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess image and keypoints."""
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original size for keypoint scaling
        original_size = (image.shape[1], image.shape[0])  # width, height

        # Load keypoints
        with open(self.annotation_paths[idx], 'r') as f:
            annotation = json.load(f)

        keypoints = []
        for name in self.keypoint_names:
            kp = annotation["keypoints"][name]
            keypoints.extend([kp["x"], kp["y"]])
        keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        # Resize image and adjust keypoints
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        # Scale keypoints to match resized image
        scale_x = self.image_size[0] / original_size[0]
        scale_y = self.image_size[1] / original_size[1]
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Convert to tensor format
        image = torch.FloatTensor(np.transpose(image, (2, 0, 1))) / 255.0
        keypoints = torch.FloatTensor(keypoints.flatten())

        return image, keypoints


class Trainer:
    """Handles the training process."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            save_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.best_loss = float('inf')

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self, num_epochs: int) -> None:
        """Train the model for specified number of epochs."""
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for images, keypoints in self.train_loader:
                images = images.to(self.device)
                keypoints = keypoints.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, keypoints)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, keypoints in self.val_loader:
                    images = images.to(self.device)
                    keypoints = keypoints.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, keypoints)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)

            # Log progress
            logging.info(
                f'Epoch [{epoch + 1}/{num_epochs}] '
                f'Train Loss: {train_loss:.4f} '
                f'Val Loss: {val_loss:.4f}'
            )

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    },
                    self.save_dir / 'best_model.pth'
                )
                logging.info(f'Saved new best model with validation loss: {val_loss:.4f}')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Set paths
    images_dir = Path("../dataset/training/images/game_1")
    annotations_dir = Path("annotations")
    save_dir = Path("saved_models")

    # Create dataset
    dataset = PingPongTableDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        max_samples=15
    )

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize model, loss, and optimizer
    model = KeypointModel(num_keypoints=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir
    )

    # Train model
    trainer.train(num_epochs=300)


if __name__ == "__main__":
    main()