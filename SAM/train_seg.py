import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import UNet
import os


def train(image_dir, mask_dir, output_dir, epochs=100, batch_size=8, learning_rate=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    dataloader = get_dataloader(image_dir, mask_dir, batch_size=batch_size)
    print(f"Loaded {len(dataloader.dataset)} samples.")

    # Model
    model = UNet()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "table_segmentation_model.pth"))
    print("Model saved successfully!")


if __name__ == "__main__":
    image_dir = "dataset/table_segmentation/images"
    mask_dir = "dataset/table_segmentation/masks"
    output_dir = "output"

    train(image_dir, mask_dir, output_dir)
