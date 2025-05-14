import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_transform(train=True):
    t = []
    t.append(transforms.ToTensor())
    if train:
        t.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(t)

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def get_voc_dataloaders(data_dir, batch_size=4, num_workers=0):
    print("Attempting to load VOC train dataset...")
    train_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='trainval', download=True, transform=get_transform(train=True)
    )
    print(f"Successfully loaded train dataset with {len(train_dataset)} images.")
    print("Attempting to load VOC val dataset...")
    val_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='val', download=True, transform=get_transform(train=False))
    print(f"Successfully loaded val dataset with {len(val_dataset)} images.")

    print("Creating DataLoaders with num_workers=0...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    print("DataLoaders created successfully.")

    # Example of accessing one item to check if loading works
    try:
        print("Attempting to access first item from train_loader...")
        images, targets = next(iter(train_loader))
        print(f"Successfully accessed first batch from train_loader with {len(images)} images.")
        # Optional: Print keys of the first target dictionary to verify structure
        if targets and isinstance(targets[0], dict): print(f"First target dictionary keys: {list(targets[0].keys())}")
        elif targets: print(f"First item in targets is not a dict, its type is {type(targets[0])}")
        else: print("Targets list is empty.")
    except Exception as e:
        print(f"Error accessing first item from train_loader: {e}")

    return train_loader, val_loader 