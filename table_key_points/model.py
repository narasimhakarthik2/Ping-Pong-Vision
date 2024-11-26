# model.py
import torch
import torch.nn as nn
from torchvision import models


class KeypointModel(nn.Module):
    def __init__(self, num_keypoints: int):
        super().__init__()
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features

        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)  # *2 for x,y coordinates
        )

    def forward(self, x):
        return self.backbone(x)