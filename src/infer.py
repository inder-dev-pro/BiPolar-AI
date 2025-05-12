import torch
from PIL import Image
import torchvision.transforms as T
import sys
import argparse
from models.model import get_model
from utils import plot_image_with_boxes

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    return transform(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='./outputs/model_epoch_10.pth', help='Path to model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21
    model = get_model(num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    img = load_image(args.image_path)
    with torch.no_grad():
        prediction = model([img.to(device)])
    boxes = prediction[0]['boxes'].cpu().numpy()
    plot_image_with_boxes(img, boxes)

if __name__ == '__main__':
    main() 