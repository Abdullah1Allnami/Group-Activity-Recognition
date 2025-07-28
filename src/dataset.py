import os
import torch
from PIL import Image
from utils import custom_collate_fn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class VolleyballDataset(Dataset):
    def __init__(self, video_ids, data_root_dir):
        self.video_ids = video_ids
        self.data_root_dir = data_root_dir
        self.samples = self.load_samples()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_samples(self):
        samples = []
        for vid in self.video_ids:
            ann_file = os.path.join(self.data_root_dir, vid, "annotations.txt")
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split()
                    frame_id = parts[0].split(".")[0]
                    activity = parts[1]
                    players_annotations = parts[2:]

                    players = []
                    for i in range(0, len(players_annotations), 5):
                        x, y, w, h, action = players_annotations[i : i + 5]
                        players.append(
                            {"action": action, "bbox": [int(x), int(y), int(w), int(h)]}
                        )

                    samples.append(
                        {
                            "video_id": vid,
                            "frame_id": frame_id,
                            "activity": activity,
                            "players": players,
                        }
                    )
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = os.path.join(self.data_root_dir, sample["video_id"])
        frame_path = os.path.join(video_path, sample["frame_id"])
        frame_id = int(sample["frame_id"])

        img_path = os.path.join(frame_path, f"{frame_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Player bounding boxes and actions
        players = sample["players"]
        player_imgs = []
        player_actions = []
        for p in players:
            x, y, w, h = p["bbox"]
            cropped = img.crop((x, y, x + w, y + h))
            if self.transform:
                cropped = self.transform(cropped)
            player_imgs.append(cropped)
            player_actions.append(p["action"])

        if self.transform:
            img = self.transform(img)

        return {
            "frame": img,
            "activity": sample["activity"],
            "player_imgs": player_imgs,
            "player_actions": player_actions,
        }

    def __len__(self):
        return len(self.samples)


def get_data_loaders(data_root_dir, batch_size):
    train_ids = [
        "1",
        "3",
        "6",
        "7",
        "10",
        "13",
        "15",
        "16",
        "18",
        "22",
        "23",
        "31",
        "32",
        "36",
        "38",
        "39",
        "40",
        "41",
        "42",
        "48",
        "50",
        "52",
        "53",
        "54",
    ]
    validation_ids = [
        "0",
        "2",
        "8",
        "12",
        "17",
        "19",
        "24",
        "26",
        "27",
        "28",
        "30",
        "33",
        "46",
        "49",
        "51",
    ]
    test_ids = [
        "4",
        "5",
        "9",
        "11",
        "14",
        "20",
        "21",
        "25",
        "29",
        "34",
        "35",
        "37",
        "43",
        "44",
        "45",
        "47",
    ]

    train_data = VolleyballDataset(train_ids, data_root_dir)
    validation_data = VolleyballDataset(validation_ids, data_root_dir)
    test_data = VolleyballDataset(test_ids, data_root_dir)

    train_dl = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_dl = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_dl = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    return train_dl, val_dl, test_dl
