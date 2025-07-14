import torch


def custom_collate_fn(batch):
    frames = torch.stack([item["frame"] for item in batch])
    activities = [item["activity"] for item in batch]
    player_imgs = [item["player_imgs"] for item in batch]
    player_actions = [item["player_actions"] for item in batch]

    return {
        "frame": frames,
        "activity": activities,
        "player_imgs": player_imgs,
        "player_actions": player_actions,
    }
