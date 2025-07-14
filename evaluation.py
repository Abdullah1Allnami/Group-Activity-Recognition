import torch
import torch.nn as nn


def test_model(model, test_dl, device, activity2idx, action2idx):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_dl:
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

            outputs = model(frame, player_image, player_actions)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    avg_loss = total_loss / len(test_dl)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")
