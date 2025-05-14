import torch
import torchvision
import os
from models.model import get_model
from dataset import get_voc_dataloaders
from tqdm import tqdm
import torch_directml # Import torch_directml

def main():
    # Check for DirectML (AMD/Intel/Qualcomm GPU on Windows) first
    try:
        device = torch_directml.device()
        print(f"Using device: {device}")
    except Exception as e:
        print(f"DirectML device not available: {e}. Falling back to CPU.")
        # Fallback to CUDA if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    num_classes = 21
    model = get_model(num_classes)
    model.to(device)

    data_dir = './data'
    # Ensure num_workers is compatible with DirectML/CPU. 0 is safest for debugging.
    # If training works with 0, you can try increasing it, but multiprocessing with DirectML can sometimes have issues.
    train_loader, val_loader = get_voc_dataloaders(data_dir, batch_size=4, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    os.makedirs('./outputs', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move images and targets to the selected device
            images = list(img.to(device) for img in images)
            
            processed_targets = []
            for t in targets:
                if t is None:
                     processed_targets.append({'boxes': torch.empty((0, 4), dtype=torch.float32, device=device), 'labels': torch.empty((0,), dtype=torch.int64, device=device)})
                else:
                    target_dict = {}
                    target_dict['boxes'] = t.get('boxes', torch.empty((0, 4), dtype=torch.float32)).to(device)
                    target_dict['labels'] = t.get('labels', torch.empty((0,), dtype=torch.int64)).to(device)
                    for key in t.keys():
                        if key not in ['boxes', 'labels']:
                             if isinstance(t[key], torch.Tensor):
                                 target_dict[key] = t[key].to(device)
                             else:
                                 target_dict[key] = t[key]
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