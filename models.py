import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet101, ResNet101_Weights


class ActivityModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  #

        self.conv_block = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, frame, player_img, player_action):
        x = self.backbone(frame)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x
