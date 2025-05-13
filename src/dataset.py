import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as F

def get_transform(train):
    transforms = []
    transforms.append(F.ToTensor())
    return F.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_voc_dataloaders(data_dir, batch_size=4, num_workers=2):
    train_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='trainval', download=True, transform=get_transform(train=True)
    )
    val_dataset = datasets.VOCDetection(
        data_dir, year='2007', image_set='val', download=True, transform=get_transform(train=False)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader 