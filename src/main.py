from dataset import get_data_loaders
from train import train_base_line_1
import torch
from evaluation import test_model

activity2idx = {
    "r_set": 0,
    "r-set": 0,
    "r_spike": 1,
    "r-spike": 1,
    "r_pass": 2,
    "r-pass": 2,
    "r_winpoint": 3,
    "r-winpoint": 3,
    "l_winpoint": 4,
    "l-winpoint": 4,
    "l_pass": 5,
    "l-pass": 5,
    "l_spike": 6,
    "l-spike": 6,
    "l_set": 7,
    "l-set": 7,
}


action2idx = {
    "waiting": 0,
    "setting": 1,
    "digging": 2,
    "falling": 3,
    "spiking": 4,
    "blocking": 5,
    "jumping": 6,
    "moving": 7,
    "standing": 8,
}


def main():
    print("*" * 20, "Main", "*" * 20)

    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    num_epochs = 20

    # Data Loading
    data_root_dir = "/kaggle/input/volleyball/volleyball_/videos"
    train_dl, val_dl, test_dl = get_data_loaders(data_root_dir, batch_size)

    # Model Training BaseLine1
    model = train_base_line_1(
        train_dl, val_dl, num_epochs, device, activity2idx, action2idx
    )
    torch.save(model.state_dict(), "activity_model.pth")
    test_model(model, test_dl, device, action2idx, action2idx)

    print("*" * 20, "RETURN", "*" * 20)


if __name__ == "__main__":
    main()
