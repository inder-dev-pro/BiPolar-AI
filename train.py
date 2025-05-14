import torch
import torchvision
import os
from models.model import get_model
from dataset import get_voc_dataloaders
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21
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
            # Ensure targets have expected keys even if empty
            processed_targets = []
            for t in targets:
                if t is None: # Should be handled by collate_fn, but as a fallback
                    processed_targets.append({'boxes': torch.empty((0, 4), dtype=torch.float32, device=device), 'labels': torch.empty((0,), dtype=torch.int64, device=device)})
                else:
                    # Assuming t is a dictionary, add missing keys with empty tensors if needed
                    # This is a simplified check; a more robust solution might inspect expected keys
                    target_dict = {}
                    target_dict['boxes'] = t.get('boxes', torch.empty((0, 4), dtype=torch.float32)).to(device)
                    target_dict['labels'] = t.get('labels', torch.empty((0,), dtype=torch.int64)).to(device)
                    # Copy other keys if they exist, like 'image_id', 'area', 'iscrowd'
                    for key in t.keys():
                        if key not in ['boxes', 'labels']:
                             # Need to handle moving different tensor types to device appropriately
                             target_dict[key] = t[key].to(device) if isinstance(t[key], torch.Tensor) else t[key]
                    processed_targets.append(target_dict)

            loss_dict = model(images, processed_targets)
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