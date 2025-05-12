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

def get_voc_dataloaders(data_dir, batch_size=4, num_workers=2):
    train_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='trainval', download=True, transform=get_transform(train=True)
    )
    val_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='val', download=True, transform=get_transform(train=False)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    return train_loader, val_loader 