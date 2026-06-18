import torch
from torch import nn
from torchvision import models


class GroupActivityRecognition(nn.Module):
    """
    Standard ResNet-50 model:
    - Backbone (ResNet-50) directly extracts features and performs Group Activity classification.
    """
    def __init__(self, num_group_classes=8, num_action_classes=9, embed_dim=512, dropout=0.3):
        super(GroupActivityRecognition, self).__init__()
        # Load ResNet-50 backbone
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.resnet = models.resnet50(pretrained=True)
            
        # Replace the final fully connected layer of ResNet-50 with dropout and linear layer
        in_features = self.resnet.fc.in_features
        if dropout > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_group_classes)
            )
        else:
            self.resnet.fc = nn.Linear(in_features, num_group_classes)
        
        # Keep num_action_classes to construct empty action predictions of the right shape
        self.num_action_classes = num_action_classes

    def forward(self, images, player_boxes=None, player_counts=None):
        """
        images: tensor of size (B, C, H, W)
        player_boxes: list of tensors (unused, kept for API compatibility)
        player_counts: list of integers (unused, kept for API compatibility)
        """
        group_output = self.resnet(images)
        # Return empty action predictions for compatibility with training/validation loops
        action_output = torch.zeros(0, self.num_action_classes, device=images.device)
        return group_output, action_output
