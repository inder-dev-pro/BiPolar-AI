import torch
import os
from models.model import get_model
from dataset import get_voc_dataloaders
from utils import plot_image_with_boxes
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21
    model = get_model(num_classes)
    model.load_state_dict(torch.load('./outputs/model_epoch_10.pth', map_location=device))
    model.to(device)
    model.eval()

    data_dir = './data'
    _, val_loader = get_voc_dataloaders(data_dir, batch_size=1, num_workers=2)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(val_loader, desc='Evaluating')):
            images = [img.to(device) for img in images]
            outputs = model(images)
            img = images[0].cpu()
            boxes = outputs[0]['boxes'].cpu().numpy()
            plot_image_with_boxes(img, boxes)
            if idx >= 4:
                break
    print('Evaluation complete. For full mAP evaluation, use torchvision references or external tools.')

if __name__ == '__main__':
    main() 