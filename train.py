import numpy as np
import torch
from tqdm import tqdm

from utils import compute_macro_f1


def train_epoch(model, dataloader, criterion_group, optimizer, device, scheduler=None):
    model.train()
    epoch_group_loss = 0.0
    
    group_correct = 0
    group_total = 0
    
    all_group_preds = []
    all_group_labels = []
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, annotations in loop:
        images = images.to(device)
        
        # Prepare targets
        batch_group_labels = []
        for ann in annotations:
            batch_group_labels.append(ann['groupLabel_idx'])
            
        group_labels = torch.tensor(batch_group_labels, dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        group_outputs, _ = model(images, annotations)
        
        # Calculate loss
        loss = criterion_group(group_outputs, group_labels)
            
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track group prediction metrics
        _, pred_group = torch.max(group_outputs, 1)
        group_correct += (pred_group == group_labels).sum().item()
        group_total += group_labels.size(0)
        
        all_group_preds.extend(pred_group.cpu().numpy())
        all_group_labels.extend(group_labels.cpu().numpy())
        
        epoch_group_loss += loss.item() * images.size(0)
        
        # Display loss inside tqdm bar
        loop.set_postfix(loss=loss.item(), group_acc=group_correct/max(group_total, 1))
        
    num_samples = len(dataloader.dataset)
    group_f1 = compute_macro_f1(np.array(all_group_preds), np.array(all_group_labels))
    return {
        'loss': epoch_group_loss / num_samples,
        'acc': (group_correct / group_total) if group_total > 0 else 0.0,
        'f1': group_f1
    }


@torch.no_grad()
def val_epoch(model, dataloader, criterion_group, device):
    model.eval()
    epoch_group_loss = 0.0
    
    group_correct = 0
    group_total = 0
    
    all_group_preds = []
    all_group_labels = []
    
    for images, annotations in dataloader:
        images = images.to(device)
        
        batch_group_labels = []
        for ann in annotations:
            batch_group_labels.append(ann['groupLabel_idx'])
            
        group_labels = torch.tensor(batch_group_labels, dtype=torch.long, device=device)
        group_outputs, _ = model(images, annotations)
        
        # Calculate loss
        loss = criterion_group(group_outputs, group_labels)
            
        # Group metrics
        _, pred_group = torch.max(group_outputs, 1)
        group_correct += (pred_group == group_labels).sum().item()
        group_total += group_labels.size(0)
        
        all_group_preds.extend(pred_group.cpu().numpy())
        all_group_labels.extend(group_labels.cpu().numpy())
        
        epoch_group_loss += loss.item() * images.size(0)
        
    num_samples = len(dataloader.dataset)
    group_f1 = compute_macro_f1(np.array(all_group_preds), np.array(all_group_labels))
    return {
        'loss': epoch_group_loss / num_samples,
        'acc': (group_correct / group_total) if group_total > 0 else 0.0,
        'f1': group_f1
    }
