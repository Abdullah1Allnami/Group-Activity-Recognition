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
    def __init__(self, num_group_classes=8, num_action_classes=9, embed_dim=2048, dropout=0.3, pooling='mixed', crop_size=(224, 224)):
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
        
        # Freeze early layers of ResNet-50 (conv1, bn1, layer1, layer2)
        for name, param in self.resnet.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
                
        # Classifier input dimension based on pooling type
        feature_dim = 4096 if self.pooling == 'mixed' else 2048
        
        # Classifier for group activities fed by pooled features
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_group_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_group_classes)

    def forward(self, images, annotations=None):
        """
        images: tensor of size (B, C, H, W)
        annotations: list of dicts (length B) containing player bounding boxes
        """
        batch_size, C, H, W = images.shape
        device = images.device
        
        # Fallback if no annotations are provided
        if annotations is None:
            # Standard image forward pass: resize whole image to crop_size and extract features
            resized_images = F.interpolate(images, size=self.crop_size, mode='bilinear', align_corners=False)
            features = self.resnet(resized_images)  # (B, 2048)
            if self.pooling == 'mixed':
                features = torch.cat([features, features], dim=1)  # (B, 4096)
            group_output = self.classifier(features)
            action_output = torch.zeros(0, self.num_action_classes, device=device)
            return group_output, action_output

        crop_list = []
        batch_indices = []
        
        # Extract crops for each player in each image in the batch
        for i in range(batch_size):
            player_anns = annotations[i].get('playersAnnotations', [])
            img = images[i]  # (C, H, W)
            
            for player in player_anns:
                x = player['x']
                y = player['y']
                w = player['w']
                h = player['h']
                
                # Clamp bounding box coordinates to image boundaries
                x1 = max(0, min(x, W - 1))
                y1 = max(0, min(y, H - 1))
                x2 = max(x1 + 1, min(x + w, W))
                y2 = max(y1 + 1, min(y + h, H))
                
                crop = img[:, y1:y2, x1:x2]
                
                # Resize crop to (224, 224)
                crop_resized = F.interpolate(
                    crop.unsqueeze(0),
                    size=self.crop_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Crop-Level Augmentations (Flip & Cutout) during training
                if self.training:
                    # Random Horizontal Flip
                    if torch.rand(1).item() < 0.5:
                        crop_resized = torch.flip(crop_resized, dims=[2])
                    # Cutout (Random Erasing)
                    if torch.rand(1).item() < 0.2:
                        ch, cw = self.crop_size
                        erase_h = torch.randint(20, 50, (1,)).item()
                        erase_w = torch.randint(20, 50, (1,)).item()
                        y_start = torch.randint(0, ch - erase_h, (1,)).item()
                        x_start = torch.randint(0, cw - erase_w, (1,)).item()
                        crop_resized[:, y_start:y_start+erase_h, x_start:x_start+erase_w] = 0.0
                
                crop_list.append(crop_resized)
                batch_indices.append(i)
                
        if len(crop_list) == 0:
            # If no crops were extracted, fall back to whole image features as baseline
            resized_images = F.interpolate(images, size=self.crop_size, mode='bilinear', align_corners=False)
            pooled_features = self.resnet(resized_images)  # (B, 2048)
            if self.pooling == 'mixed':
                pooled_features = torch.cat([pooled_features, pooled_features], dim=1)  # (B, 4096)
        else:
            # Stack all crops into a single tensor for a single efficient backbone pass
            crops_tensor = torch.stack(crop_list, dim=0)  # (M, C, 224, 224)
            
            # Extract 2048 features for each crop
            crop_features = self.resnet(crops_tensor)  # (M, 2048)
            
            # Feature-Level L2 Normalization
            crop_features = F.normalize(crop_features, p=2, dim=1)
            
            # Pool features over all people for each frame in the batch
            pooled_features_list = []
            batch_indices = torch.tensor(batch_indices, device=device)
            
            for i in range(batch_size):
                mask = (batch_indices == i)
                if mask.any():
                    img_crop_features = crop_features[mask]  # (N_i, 2048)
                    if self.pooling == 'max':
                        pooled, _ = torch.max(img_crop_features, dim=0)  # (2048,)
                    elif self.pooling in ('mean', 'avg'):
                        pooled = torch.mean(img_crop_features, dim=0)  # (2048,)
                    elif self.pooling == 'mixed':
                        max_pooled, _ = torch.max(img_crop_features, dim=0)
                        avg_pooled = torch.mean(img_crop_features, dim=0)
                        pooled = torch.cat([max_pooled, avg_pooled], dim=0)  # (4096,)
                    else:
                        pooled, _ = torch.max(img_crop_features, dim=0)
                else:
                    # Fallback if no crops detected for this specific frame
                    feat_dim = 4096 if self.pooling == 'mixed' else 2048
                    pooled = torch.zeros(feat_dim, device=device)
                pooled_features_list.append(pooled)
                
            pooled_features = torch.stack(pooled_features_list, dim=0)  # (B, feature_dim)
            
        group_output = self.classifier(pooled_features)
        
        # Return empty action predictions for compatibility with training/validation loops
        action_output = torch.zeros(0, self.num_action_classes, device=device)
        return group_output, action_output
