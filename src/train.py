import torch
import torchvision
import os
from models.model import get_model
from dataset import get_voc_dataloaders
from utils import collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21  # 20 classes + background for Pascal VOC
    model = get_model(num_classes)
    model.to(device)

    data_dir = './data'
    train_loader, val_loader = get_voc_dataloaders(data_dir, batch_size=4, num_workers=2)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    os.makedirs('./outputs', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = list(img.to(device) for img in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        lr_scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}')
        torch.save(model.state_dict(), f'./outputs/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 