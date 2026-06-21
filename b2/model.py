import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class GroupActivityRecognitionB2(nn.Module):
    """
    B2: Fine-tuned Person Classification
    - The ResNet-50 CNN model is deployed on each person crop.
    - Feature extraction for each crop (2048 features) is pooled over all people.
    - Pooled features are fed to a softmax classifier to recognize group activities.
    """
    def __init__(self, num_group_classes=8, num_action_classes=9, embed_dim=2048, dropout=0.3, pooling='max', crop_size=(224, 224)):
        super(GroupActivityRecognitionB2, self).__init__()
        self.num_group_classes = num_group_classes
        self.num_action_classes = num_action_classes
        self.pooling = pooling.lower()
        self.crop_size = crop_size
        
        # Load ResNet-50 backbone
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.resnet = models.resnet50(pretrained=True)
            
        # Replace the final fully connected layer of ResNet-50 with Identity to extract 2048-dim features
        self.resnet.fc = nn.Identity()
        
        # Classifier for group activities fed by pooled 2048-dim features
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, num_group_classes)
            )
        else:
            self.classifier = nn.Linear(2048, num_group_classes)

    def forward(self, images, annotations=None):
        """
        images: tensor of size (B, C, H, W)
        annotations: list of dicts (length B) containing player bounding boxes
        """
        batch_size, C, H, W = images.shape
        device = images.device
        
        if annotations is None:
            features = self.resnet(images)
            group_output = self.classifier(features)
            action_output = torch.zeros(0, self.num_action_classes, device=device)
            return group_output, action_output

        crop_list = []
        batch_indices = []
        
        # Extract crops for each player in each image in the batch
        for i in range(batch_size):
            player_anns = annotations[i].get('playersAnnotations', [])
            img = images[i]
            
            for player in player_anns:
                x = player['x']
                y = player['y']
                w = player['w']
                h = player['h']
                
                x1 = max(0, min(x, W - 1))
                y1 = max(0, min(y, H - 1))
                x2 = max(x1 + 1, min(x + w, W))
                y2 = max(y1 + 1, min(y + h, H))
                
                crop = img[:, y1:y2, x1:x2]
                
                crop_resized = F.interpolate(
                    crop.unsqueeze(0),
                    size=self.crop_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                crop_list.append(crop_resized)
                batch_indices.append(i)
                
        if len(crop_list) == 0:
            pooled_features = self.resnet(images)
        else:
            crops_tensor = torch.stack(crop_list, dim=0)
            crop_features = self.resnet(crops_tensor)
            
            pooled_features_list = []
            batch_indices = torch.tensor(batch_indices, device=device)
            
            for i in range(batch_size):
                mask = (batch_indices == i)
                if mask.any():
                    img_crop_features = crop_features[mask]
                    if self.pooling == 'max':
                        pooled, _ = torch.max(img_crop_features, dim=0)
                    else:
                        pooled = torch.mean(img_crop_features, dim=0)
                else:
                    pooled = torch.zeros(2048, device=device)
                pooled_features_list.append(pooled)
                
            pooled_features = torch.stack(pooled_features_list, dim=0)
            
        group_output = self.classifier(pooled_features)
        
        # Return empty action predictions for compatibility with training/validation loops
        action_output = torch.zeros(0, self.num_action_classes, device=device)
        return group_output, action_output
