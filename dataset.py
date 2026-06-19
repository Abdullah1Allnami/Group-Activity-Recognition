import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import GROUP_ACTIVITIES, PLAYER_ACTIONS, group_to_idx, action_to_idx


class VolleyBallDataset(Dataset):
    def __init__(self, split='train', transform=None, data_path='./volleyball/volleyball_/videos', seq_len=None, stride=3):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.seq_len = seq_len
        self.stride = stride
        
        # Define video splits
        self.train_videos = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
        self.val_videos = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
        self.test_videos = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
        
        self.data = []
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' not found.")
            
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
        
        # Sequence mode
        if self.seq_len is not None and self.seq_len > 1:
            # We are reading from disk
            frame_dir = os.path.dirname(image_frame)
            frame_name = os.path.basename(image_frame).split('.')[0]
            
            # List all frames inside the directory and sort them chronologically
            all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
            
            # Find the annotated frame index in the list of frames
            target_file = os.path.basename(image_frame)
            if target_file in all_frames:
                mid_idx = all_frames.index(target_file)
            else:
                mid_idx = len(all_frames) // 2
                
            # Sample seq_len frames centered around mid_idx
            half_len = self.seq_len // 2
            offsets = [i * self.stride for i in range(-half_len, half_len + 1)]
            
            images = []
            for offset in offsets:
                sample_idx = mid_idx + offset
                # Clamp to index boundaries [0, len(all_frames) - 1]
                sample_idx = max(0, min(sample_idx, len(all_frames) - 1))
                frame_path = os.path.join(frame_dir, all_frames[sample_idx])
                
                image = cv2.imread(frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

            # Scale annotations based on the first frame's original size
            h_orig, w_orig = images[0].shape[:2]
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
            group_idx = group_to_idx.get(group_str.lower(), 0) if not str(group_str).isdigit() else int(group_str)
            
            scaled_annotation = {
                'groupLabel': group_str,
                'groupLabel_idx': group_idx,
                'playersAnnotations': scaled_players
            }
            
            # Apply transformation to each frame
            processed_images = []
            for img in images:
                if h_orig != target_h or w_orig != target_w:
                    img = cv2.resize(img, (target_w, target_h))
                    
                if self.transform:
                    from PIL import Image
                    img = Image.fromarray(img)
                    img = self.transform(img)
                else:
                    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                processed_images.append(img)
                
            # Stack images along sequence dimension: (seq_len, C, H, W)
            images_tensor = torch.stack(processed_images, dim=0)
            
            return images_tensor, scaled_annotation

        # Single frame mode (seq_len is None or 1)
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
