import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class ActivityModel(nn.Module):
    def __init__(self, num_classes, num_action_classes=9):
        super().__init__()

        # Backbone for full frame
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, H, W]

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

        # For player crops
        player_resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.player_backbone = nn.Sequential(*list(player_resnet.children())[:-2])
        for param in self.player_backbone.parameters():
            param.requires_grad = False

        self.player_feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
        )

        # Embedding for action classes
        self.action_embedding = nn.Embedding(
            num_embeddings=num_action_classes, embedding_dim=32
        )

        # Classifier that takes frame feature + aggregated player features
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, frame, player_imgs_batch, player_actions_batch):
        B = frame.size(0)

        # Full-frame feature
        x = self.backbone(frame)  # [B, 2048, H, W]
        x = self.conv_block(x)  # [B, 128, 1, 1]
        x = x.view(B, -1)  # [B, 128]

        device = frame.device
        batch_player_features = []

        for i in range(B):
            player_imgs = player_imgs_batch[i]  # List of [3, H, W]
            player_actions = player_actions_batch[i]  # List of action indices

            feats = []
            for img, act in zip(player_imgs, player_actions):
                img = img.unsqueeze(0).to(device)  # [1, 3, H, W]
                act = torch.tensor([act], device=device)

                img_feat = self.player_backbone(img)  # [1, 2048, H, W]
                img_feat = self.player_feature_head(img_feat)  # [1, 128]
                act_embed = self.action_embedding(act)  # [1, 32]

                combined = torch.cat([img_feat, act_embed], dim=1)  # [1, 160]
                feats.append(combined)

            if feats:
                feats = torch.cat(feats, dim=0)  # [N, 160]
                agg_feat = feats.mean(dim=0, keepdim=True)  # [1, 160]
            else:
                agg_feat = torch.zeros(1, 160, device=device)

            batch_player_features.append(agg_feat)

        player_feats = torch.cat(batch_player_features, dim=0)  # [B, 160]
        combined = torch.cat([x, player_feats], dim=1)  # [B, 288]
        out = self.classifier(combined)  # [B, num_classes]
        return out
