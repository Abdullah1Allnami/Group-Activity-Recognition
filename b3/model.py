import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class GroupActivityRecognitionB3(nn.Module):
    """
    B3: Temporal Model with Image Features
    - Backbone (ResNet-50) extracts features for each of the 9 frames in a clip.
    - An LSTM processes the sequence of 9 frame features.
    - The output of the LSTM (at the final step) is fed to a classifier to recognize group activities.
    """
    def __init__(self, num_group_classes=8, num_action_classes=9, embed_dim=2048, hidden_size=512, num_layers=1, dropout=0.3):
        super(GroupActivityRecognitionB3, self).__init__()
        self.num_group_classes = num_group_classes
        self.num_action_classes = num_action_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Load ResNet-50 backbone
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.resnet = models.resnet50(pretrained=True)
            
        # Replace the final fully connected layer of ResNet-50 with Identity to extract 2048-dim features
        self.resnet.fc = nn.Identity()
        
        # LSTM layer to model temporal sequence of image features
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier for group activities fed by the final timestep output of the LSTM
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, num_group_classes)
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_group_classes)

    def forward(self, images, annotations=None):
        """
        images: tensor of size (B, seq_len, C, H, W)
        annotations: list of dicts (unused, kept for API compatibility)
        """
        batch_size, seq_len, C, H, W = images.shape
        device = images.device
        
        # Reshape to (B * seq_len, C, H, W) for batch computation through backbone
        images_flat = images.view(batch_size * seq_len, C, H, W)
        
        # Extract features for all frames in the batch
        features_flat = self.resnet(images_flat)  # (B * seq_len, 2048)
        
        # Reshape back to sequence format: (B, seq_len, 2048)
        features = features_flat.view(batch_size, seq_len, -1)
        
        # Pass sequence features through LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (B, seq_len, hidden_size)
        
        # Select output from the final timestep
        out = lstm_out[:, -1, :]  # (B, hidden_size)
        
        # Classify group activity
        group_output = self.classifier(out)
        
        # Return empty action predictions for compatibility with training/validation loops
        action_output = torch.zeros(0, self.num_action_classes, device=device)
        return group_output, action_output
