import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

class IoUTracker:
    def __init__(self, use_detector=True, detector_threshold=0.5):
        self.use_detector = use_detector
        self.detector_threshold = detector_threshold
        self.detector = None

    def _get_detector(self, device):
        if self.detector is None and self.use_detector:
            try:
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device)
                self.detector.eval()
            except Exception as e:
                print(f"Warning: Failed to load pretrained detector: {e}. Falling back to constant box propagation.")
                self.use_detector = False
        return self.detector

    def track(self, images, annotations, device):
        """
        images: (B, seq_len, C, H, W)
        annotations: list of B dicts containing 'playersAnnotations' for the middle frame
        device: torch.device
        
        Returns:
            tracked_boxes_batch: list of length B, where each item is a tensor of shape (seq_len, num_players, 4)
        """
        batch_size, seq_len, C, H, W = images.shape
        mid_idx = seq_len // 2
        tracked_boxes_batch = []
        
        detector = self._get_detector(device)

        for i in range(batch_size):
            player_anns = annotations[i].get('playersAnnotations', [])
            num_players = len(player_anns)
            
            # Extract middle frame boxes
            mid_boxes = []
            for player in player_anns:
                x, y, w, h = player['x'], player['y'], player['w'], player['h']
                x1 = max(0, min(x, W - 1))
                y1 = max(0, min(y, H - 1))
                x2 = max(x1 + 1, min(x + w, W))
                y2 = max(y1 + 1, min(y + h, H))
                mid_boxes.append([x1, y1, x2, y2])
            
            if len(mid_boxes) == 0:
                # Fallback if no players annotated
                mid_boxes = [[0, 0, W, H]]
                num_players = 1

            mid_boxes_tensor = torch.tensor(mid_boxes, dtype=torch.float32, device=device)
            seq_boxes = torch.zeros((seq_len, num_players, 4), dtype=torch.float32, device=device)
            seq_boxes[mid_idx] = mid_boxes_tensor

            if not self.use_detector or detector is None:
                # Constant box propagation fallback
                for t in range(seq_len):
                    seq_boxes[t] = mid_boxes_tensor
            else:
                # Track forward in time: mid_idx -> mid_idx + 1 -> ...
                for t in range(mid_idx + 1, seq_len):
                    prev_boxes = seq_boxes[t - 1]
                    detections = self._detect_persons(images[i, t], detector, device)
                    if len(detections) == 0:
                        seq_boxes[t] = prev_boxes
                    else:
                        seq_boxes[t] = self._match_boxes(prev_boxes, detections)

                # Track backward in time: mid_idx -> mid_idx - 1 -> ...
                for t in range(mid_idx - 1, -1, -1):
                    next_boxes = seq_boxes[t + 1]
                    detections = self._detect_persons(images[i, t], detector, device)
                    if len(detections) == 0:
                        seq_boxes[t] = next_boxes
                    else:
                        seq_boxes[t] = self._match_boxes(next_boxes, detections)

            tracked_boxes_batch.append(seq_boxes)

        return tracked_boxes_batch

    @torch.no_grad()
    def _detect_persons(self, img, detector, device):
        """
        img: (C, H, W)
        """
        # img is scaled in [0, 1]
        outputs = detector([img])
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        
        # Class 1 is 'person' in COCO dataset used by the detector
        mask = (labels == 1) & (scores >= self.detector_threshold)
        return boxes[mask]

    def _match_boxes(self, track_boxes, detections):
        """
        track_boxes: (N, 4)
        detections: (M, 4)
        """
        iou_matrix = self._box_iou(track_boxes, detections)
        N = track_boxes.shape[0]
        matched_boxes = track_boxes.clone()
        
        iou_np = iou_matrix.cpu().numpy()
        matched_detections = set()
        
        for idx in range(N):
            best_det_idx = -1
            best_iou = 0.3  # min IoU threshold for matching
            for j in range(detections.shape[0]):
                if j in matched_detections:
                    continue
                if iou_np[idx, j] > best_iou:
                    best_iou = iou_np[idx, j]
                    best_det_idx = j
            if best_det_idx != -1:
                matched_boxes[idx] = detections[best_det_idx]
                matched_detections.add(best_det_idx)
                
        return matched_boxes

    def _box_iou(self, boxes1, boxes2):
        # area of boxes1
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        # area of boxes2
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        union = area1[:, None] + area2 - inter
        return inter / torch.clamp(union, min=1e-6)


class GroupActivityRecognitionB4(nn.Module):
    """
    B4: Temporal Model with Person Features
    - Employs a pretrained detector + IoU tracker to obtain player boxes across 9 frames.
    - Crops player regions and extracts features with a ResNet-50 backbone.
    - Runs a player-level LSTM to model the temporal sequence of each player's features.
    - Pools features across all players (max or mean) to recognize the group activity.
    """
    def __init__(self, num_group_classes=8, num_action_classes=9, embed_dim=2048, hidden_size=512, num_layers=1, dropout=0.3, pooling='max', crop_size=(224, 224), use_detector=True):
        super(GroupActivityRecognitionB4, self).__init__()
        self.num_group_classes = num_group_classes
        self.num_action_classes = num_action_classes
        self.pooling = pooling.lower()
        self.crop_size = crop_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize the tracker
        self.tracker = IoUTracker(use_detector=use_detector)
        
        # Load ResNet-50 backbone
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.resnet = models.resnet50(pretrained=True)
            
        self.resnet.fc = nn.Identity()
        
        # LSTM for temporal sequence modeling at player level
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier for group activities fed by pooled player features
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, num_group_classes)
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_group_classes)

    def forward(self, images, annotations=None):
        """
        images: (B, seq_len, C, H, W)
        annotations: list of B dicts containing player annotations for the middle frame
        """
        batch_size, seq_len, C, H, W = images.shape
        device = images.device

        if annotations is None:
            # Fallback if no annotations provided (e.g. dummy forward)
            dummy_features = torch.zeros(batch_size, self.hidden_size, device=device)
            group_output = self.classifier(dummy_features)
            action_output = torch.zeros(0, self.num_action_classes, device=device)
            return group_output, action_output

        # 1. Track player boxes across frames
        tracked_boxes_batch = self.tracker.track(images, annotations, device)

        # 2. Crop and resize player regions for all frames
        crop_list = []
        batch_indices = []
        player_counts = []
        
        for i in range(batch_size):
            seq_boxes = tracked_boxes_batch[i]  # (seq_len, num_players, 4)
            num_players = seq_boxes.shape[1]
            player_counts.append(num_players)
            
            for p in range(num_players):
                for t in range(seq_len):
                    x1, y1, x2, y2 = seq_boxes[t, p]
                    
                    x1_idx = int(max(0, min(x1.item(), W - 1)))
                    y1_idx = int(max(0, min(y1.item(), H - 1)))
                    x2_idx = int(max(x1_idx + 1, min(x2.item(), W)))
                    y2_idx = int(max(y1_idx + 1, min(y2.item(), H)))
                    
                    crop = images[i, t, :, y1_idx:y2_idx, x1_idx:x2_idx]
                    crop_resized = F.interpolate(
                        crop.unsqueeze(0),
                        size=self.crop_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    crop_list.append(crop_resized)
                    batch_indices.append((i, p))

        # 3. Extract spatial features using ResNet-50
        crops_tensor = torch.stack(crop_list, dim=0)  # (B * num_players * seq_len, C, 224, 224)
        crop_features = self.resnet(crops_tensor)     # (B * num_players * seq_len, 2048)
        
        # Reshape to (B * num_players, seq_len, 2048) for LSTM
        total_players = sum(player_counts)
        crop_features = crop_features.view(total_players, seq_len, -1)

        # 4. Process temporal crop sequences per player via LSTM
        lstm_out, (hn, cn) = self.lstm(crop_features)  # (total_players, seq_len, hidden_size)
        player_temporal_features = lstm_out[:, -1, :]  # (total_players, hidden_size)

        # 5. Pool features across individuals for each batch item
        pooled_features_list = []
        player_offset = 0
        for i in range(batch_size):
            num_players = player_counts[i]
            img_player_features = player_temporal_features[player_offset : player_offset + num_players]
            player_offset += num_players
            
            if self.pooling == 'max':
                pooled, _ = torch.max(img_player_features, dim=0)
            else:
                pooled = torch.mean(img_player_features, dim=0)
            pooled_features_list.append(pooled)

        pooled_features = torch.stack(pooled_features_list, dim=0)  # (B, hidden_size)

        # 6. Classify group activity
        group_output = self.classifier(pooled_features)
        action_output = torch.zeros(0, self.num_action_classes, device=device)
        
        return group_output, action_output
