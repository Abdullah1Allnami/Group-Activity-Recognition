import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import GROUP_ACTIVITIES, PLAYER_ACTIONS, group_to_idx, action_to_idx


class VolleyBallDataset(Dataset):
    def __init__(self, split='train', transform=None, data_path='/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos'):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Define video splits
        self.train_videos = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
        self.val_videos = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
        self.test_videos = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
        
        self.data = []
        self.load_data()

    def load_data(self):
        # Fallback to mock data if path doesn't exist
        if not os.path.exists(self.data_path):
            print(f"[Dataset Split: {self.split}] WARNING: Data path '{self.data_path}' not found.")
            print(f"-> Generating synthetic mock dataset for local verification/dry-run.")
            self.is_mock = True
            
            # Generate dummy video annotations
            num_mock_samples = 16 if self.split == 'train' else 8
            for idx in range(num_mock_samples):
                # Construct 12 players per frame
                parsed_players = []
                for p_idx in range(12):
                    # Distribute player boxes across court sides
                    x = 120 + p_idx * 85 if p_idx < 6 else 180 + (p_idx - 6) * 85
                    y = 200 + (p_idx % 3) * 120
                    w = 55
                    h = 110
                    # Add labels
                    action_str = PLAYER_ACTIONS[p_idx % len(PLAYER_ACTIONS)]
                    parsed_players.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'action': action_str
                    })
                
                groupLabel = GROUP_ACTIVITIES[idx % len(GROUP_ACTIVITIES)]
                annotation = {
                    'groupLabel': groupLabel,
                    'playersAnnotations': parsed_players
                }
                self.data.append((f"mock_frame_{idx}.jpg", annotation))
        else:
            self.is_mock = False
            for video_folder in os.listdir(self.data_path): # for each video folder
                if not video_folder.isdigit():
                    continue
                video_id = int(video_folder)
                
                # Filter by split
                if self.split == 'train' and video_id not in self.train_videos:
                    continue
                elif self.split == 'val' and video_id not in self.val_videos:
                    continue
                elif self.split == 'test' and video_id not in self.test_videos:
                    continue
                    
                video_path = os.path.join(self.data_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                annotation_file = os.path.join(video_path, 'annotations.txt')
                if not os.path.exists(annotation_file):
                    continue

                for row in open(annotation_file, 'r'):
                    frame_folder, groupLabel, *playersAnnotations = row.strip().split(' ')
                    frame_name = frame_folder.split('.')[0]
                    image_frame  = os.path.join(video_path, frame_name, frame_name + '.jpg')
                    
                    parsed_players = [{
                        'x': int(playersAnnotations[i]),
                        'y': int(playersAnnotations[i+1]),
                        'w': int(playersAnnotations[i+2]),
                        'h': int(playersAnnotations[i+3]),
                        'action': playersAnnotations[i+4]
                    } for i in range(0, len(playersAnnotations), 5)]
                    
                    annotation = {
                        'groupLabel': groupLabel,
                        'playersAnnotations': parsed_players
                    }
                    self.data.append((image_frame, annotation))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_frame, annotation = self.data[idx]
        
        # Load or generate image
        if hasattr(self, 'is_mock') and self.is_mock:
            # Create a stylized volleyball court image
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[:, :, 0] = 34  # Dark cyan green theme
            image[:, :, 1] = 139
            image[:, :, 2] = 34
            # Draw boundary court lines
            cv2.rectangle(image, (100, 50), (1180, 670), (255, 255, 255), 4)
            # Net line
            cv2.line(image, (640, 50), (640, 670), (200, 200, 200), 6)
            
            # Render mock players directly onto the image
            for p in annotation['playersAnnotations']:
                cv2.rectangle(image, (p['x'], p['y']), (p['x'] + p['w'], p['y'] + p['h']), (0, 0, 255), 3)
        else:
            image = cv2.imread(image_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        h_orig, w_orig = image.shape[:2]
        target_h, target_w = 720, 1280
        
        scale_x = target_w / w_orig
        scale_y = target_h / h_orig
        
        scaled_players = []
        for player in annotation['playersAnnotations']:
            x = int(player['x'] * scale_x)
            y = int(player['y'] * scale_y)
            w = int(player['w'] * scale_x)
            h = int(player['h'] * scale_y)
            
            action_str = player['action']
            # Map action label string to class index
            action_idx = action_to_idx.get(action_str.lower(), 0) if not str(action_str).isdigit() else int(action_str)
            
            scaled_players.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'action': action_str,
                'action_idx': action_idx
            })
            
        group_str = annotation['groupLabel']
        # Map group label string to class index
        group_idx = group_to_idx.get(group_str.lower(), 0) if not str(group_str).isdigit() else int(group_str)
        
        scaled_annotation = {
            'groupLabel': group_str,
            'groupLabel_idx': group_idx,
            'playersAnnotations': scaled_players
        }
        
        if h_orig != target_h or w_orig != target_w:
            image = cv2.resize(image, (target_w, target_h))
            
        if self.transform:
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
            
        return image, scaled_annotation
