from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
from models import ActivityModel
import torch.nn.functional as F


def base_line_1(
    train_dl,
    val_dl,
    num_epochs,
    device,
    activity2idx,
    action2idx,
    start_with_pretrain=False,
    weights_path="",
):

    num_classes = len(activity2idx)
    model = ActivityModel(num_classes).to(device)

    print(f"Training on Device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_accuracy = 0
    best_model_state = None

    if start_with_pretrain:
        print(f"Loading Trianed Weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Pretrained Weights Loaded Successfully")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_dl:
            frame = batch["frame"].to(device)
            player_image = [
                [img.to(device) for img in player_batch]
                for player_batch in batch["player_imgs"]
            ]
            player_actions = [
                [action2idx[action] for action in actions]
                for actions in batch["player_actions"]
            ]

            labels = torch.tensor(
                [activity2idx[a.strip().lower()] for a in batch["activity"]]
            ).to(device)

            optimizer.zero_grad()
            outputs = model(frame, player_image, player_actions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        # -------- Validation --------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_dl:
                frame = batch["frame"].to(device)
                player_image = [
                    [img.to(device) for img in player_batch]
                    for player_batch in batch["player_imgs"]
                ]
                player_actions = [
                    [action2idx[action] for action in actions]
                    for actions in batch["player_actions"]
                ]
                labels = torch.tensor([activity2idx[a] for a in batch["activity"]]).to(
                    device
                )

                outputs = model(frame, player_image, player_actions)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()

    # Optionally: Load the best model state
    model.load_state_dict(best_model_state)

    return model
