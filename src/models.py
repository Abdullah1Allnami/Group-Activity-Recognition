import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ActivityModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Unfreeze only the last block for fine-tuning
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.conv_block = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, frame):
        x = self.backbone(frame)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


class ActivityModel2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        for param in vit.parameters():
            param.requires_grad = False

        for param in vit.encoder.ln.parameters():
            param.requires_grad = True
        for param in vit.heads.parameters():
            param.requires_grad = True

        vit.heads = nn.Identity()

        self.backbone = vit

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
