import os
import sys
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to sys.path to allow running this file directly if executed from elsewhere
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import GROUP_ACTIVITIES, PLAYER_ACTIONS
from dataset import VolleyBallDataset
from b1.model import GroupActivityRecognition
from b2.model import GroupActivityRecognitionB2
from b3.model import GroupActivityRecognitionB3
from train import train_epoch, val_epoch


def baseline_1():
    epochs = 5
    batch_size = 4
    lr = 1e-4
    data_path = '/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos'
    
    # Override path if path doesn't exist
    if not os.path.exists(data_path):
        data_path = './volleyball/volleyball_/videos'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Baseline 1 on device: {device}")

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def collate_fn(batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        images = torch.stack(images, 0)
        return images, annotations
    
    # Create train, val, and test datasets
    train_dataset = VolleyBallDataset(split='train', transform=train_transform, data_path=data_path)
    val_dataset = VolleyBallDataset(split='val', transform=val_transform, data_path=data_path)
    test_dataset = VolleyBallDataset(split='test', transform=val_transform, data_path=data_path)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = GroupActivityRecognition(
        num_group_classes=len(GROUP_ACTIVITIES),
        num_action_classes=len(PLAYER_ACTIONS),
        embed_dim=256,
        dropout=0.3
    )
    model = model.to(device)
    
    # Differential learning rates: smaller for backbone, larger for fresh head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'resnet.fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': lr * 10, 'weight_decay': 1e-3}
    ])
    
    criterion_group = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Cosine Annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint saving paths
    os.makedirs('b1/checkpoints', exist_ok=True)
    best_model_path = 'b1/checkpoints/best_model.pth'
    final_model_path = 'b1/checkpoints/final_model.pth'
    best_val_f1 = 0.0

    # Load existing checkpoint weights before training
    if os.path.exists(best_model_path):
        print(f"Loading weights from existing best model checkpoint: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate best checkpoint: {e}")
    elif os.path.exists(final_model_path):
        print(f"Loading weights from existing final model checkpoint: {final_model_path}")
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate final checkpoint: {e}")

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Display learning rates
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Learning Rates - Backbone: {current_lrs[0]:.2e} | Head: {current_lrs[1]:.2e}")
        
        train_metrics = train_epoch(model, train_loader, criterion_group, optimizer, device)
        val_metrics = val_epoch(model, val_loader, criterion_group, device)
        
        # Step the scheduler
        scheduler.step()
        
        # Append history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_f1'].append(train_metrics['f1'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Dashboard printout
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']*100:.2f}% | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['acc']*100:.2f}% | Val   F1: {val_metrics['f1']:.4f}")

        # Checkpoint saving logic based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} with Val F1: {best_val_f1:.4f}")

    # Save final model weights
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print("Training completed successfully!")

    # Evaluate the best model checkpoint on the test set
    if os.path.exists(best_model_path):
        print(f"\nEvaluating the best model on the test set...")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_metrics = val_epoch(model, test_loader, criterion_group, device)
            print("\n--- Test Evaluation Results ---")
            print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']*100:.2f}% | Test F1: {test_metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error during test evaluation: {e}")


def baseline_2():
    epochs = 5
    batch_size = 4
    lr = 1e-4
    data_path = '/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos'
    
    # Override path if path doesn't exist
    if not os.path.exists(data_path):
        data_path = './volleyball/volleyball_/videos'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Baseline 2 on device: {device}")

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def collate_fn(batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        images = torch.stack(images, 0)
        return images, annotations
    
    # Create train, val, and test datasets
    train_dataset = VolleyBallDataset(split='train', transform=train_transform, data_path=data_path)
    val_dataset = VolleyBallDataset(split='val', transform=val_transform, data_path=data_path)
    test_dataset = VolleyBallDataset(split='test', transform=val_transform, data_path=data_path)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = GroupActivityRecognitionB2(
        num_group_classes=len(GROUP_ACTIVITIES),
        num_action_classes=len(PLAYER_ACTIONS),
        embed_dim=2048,
        dropout=0.3,
        pooling='max'
    )
    model = model.to(device)
    
    # Differential learning rates: smaller for backbone, larger for fresh head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': lr * 10, 'weight_decay': 1e-3}
    ])
    
    # Label smoothing tuned to 0.1
    criterion_group = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Cosine Annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint saving paths
    os.makedirs('b2/checkpoints', exist_ok=True)
    best_model_path = 'b2/checkpoints/best_model.pth'
    final_model_path = 'b2/checkpoints/final_model.pth'
    best_val_f1 = 0.0

    # Load existing checkpoint weights before training
    if os.path.exists(best_model_path):
        print(f"Loading weights from existing best model checkpoint: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate best checkpoint: {e}")
    elif os.path.exists(final_model_path):
        print(f"Loading weights from existing final model checkpoint: {final_model_path}")
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate final checkpoint: {e}")

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Display learning rates
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Learning Rates - Backbone: {current_lrs[0]:.2e} | Head: {current_lrs[1]:.2e}")
        
        train_metrics = train_epoch(model, train_loader, criterion_group, optimizer, device)
        val_metrics = val_epoch(model, val_loader, criterion_group, device)
        
        # Step the scheduler
        scheduler.step()
        
        # Append history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_f1'].append(train_metrics['f1'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Dashboard printout
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']*100:.2f}% | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['acc']*100:.2f}% | Val   F1: {val_metrics['f1']:.4f}")

        # Checkpoint saving logic based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} with Val F1: {best_val_f1:.4f}")

    # Save final model weights
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print("Training completed successfully!")

    # Evaluate the best model checkpoint on the test set
    if os.path.exists(best_model_path):
        print(f"\nEvaluating the best model on the test set...")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_metrics = val_epoch(model, test_loader, criterion_group, device)
            print("\n--- Test Evaluation Results ---")
            print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']*100:.2f}% | Test F1: {test_metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error during test evaluation: {e}")


def baseline_3():
    epochs = 5
    batch_size = 4
    lr = 1e-4
    data_path = '/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos'
    
    # Override path if path doesn't exist
    if not os.path.exists(data_path):
        data_path = './volleyball/volleyball_/videos'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Baseline 3 on device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def collate_fn(batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        images = torch.stack(images, 0)
        return images, annotations
    
    # Create train, val, and test datasets
    train_dataset = VolleyBallDataset(split='train', transform=train_transform, data_path=data_path, seq_len=9, stride=3)
    val_dataset = VolleyBallDataset(split='val', transform=val_transform, data_path=data_path, seq_len=9, stride=3)
    test_dataset = VolleyBallDataset(split='test', transform=val_transform, data_path=data_path, seq_len=9, stride=3)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2 if os.name == 'posix' else 0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = GroupActivityRecognitionB3(
        num_group_classes=len(GROUP_ACTIVITIES),
        num_action_classes=len(PLAYER_ACTIONS),
        embed_dim=2048,
        hidden_size=512,
        num_layers=1,
        dropout=0.3
    )
    model = model.to(device)
    
    # Differential learning rates: smaller for backbone, larger for fresh heads (LSTM and classifier)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'lstm' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': lr * 10, 'weight_decay': 1e-3}
    ])
    
    # Label smoothing tuned to 0.1
    criterion_group = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Cosine Annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint saving paths
    os.makedirs('b3/checkpoints', exist_ok=True)
    best_model_path = 'b3/checkpoints/best_model.pth'
    final_model_path = 'b3/checkpoints/final_model.pth'
    best_val_f1 = 0.0

    # Load existing checkpoint weights before training
    if os.path.exists(best_model_path):
        print(f"Loading weights from existing best model checkpoint: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate best checkpoint: {e}")
    elif os.path.exists(final_model_path):
        print(f"Loading weights from existing final model checkpoint: {final_model_path}")
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            print("Evaluating loaded model on validation split...")
            initial_val_metrics = val_epoch(model, val_loader, criterion_group, device)
            best_val_f1 = initial_val_metrics['f1']
            print(f"Loaded model validation F1: {best_val_f1:.4f}. Reset best_val_f1 to match.")
        except Exception as e:
            print(f"Warning: could not load or evaluate final checkpoint: {e}")

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Display learning rates
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Learning Rates - Backbone: {current_lrs[0]:.2e} | Head: {current_lrs[1]:.2e}")
        
        train_metrics = train_epoch(model, train_loader, criterion_group, optimizer, device)
        val_metrics = val_epoch(model, val_loader, criterion_group, device)
        
        # Step the scheduler
        scheduler.step()
        
        # Append history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_f1'].append(train_metrics['f1'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Dashboard printout
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']*100:.2f}% | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['acc']*100:.2f}% | Val   F1: {val_metrics['f1']:.4f}")

        # Checkpoint saving logic based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} with Val F1: {best_val_f1:.4f}")

    # Save final model weights
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print("Training completed successfully!")

    # Evaluate the best model checkpoint on the test set
    if os.path.exists(best_model_path):
        print(f"\nEvaluating the best model on the test set...")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_metrics = val_epoch(model, test_loader, criterion_group, device)
            print("\n--- Test Evaluation Results ---")
            print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']*100:.2f}% | Test F1: {test_metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error during test evaluation: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate group activity recognition baselines.")
    parser.add_argument('--baseline', type=int, choices=[1, 2, 3], default=1, help="Baseline model to run (1, 2 or 3)")
    args = parser.parse_args()

    if args.baseline == 1:
        baseline_1()
    elif args.baseline == 2:
        baseline_2()
    elif args.baseline == 3:
        baseline_3()
