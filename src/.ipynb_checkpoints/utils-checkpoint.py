import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def collate_fn(batch):
    return tuple(zip(*batch))

def plot_image_with_boxes(img, boxes, labels=None):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        draw.rectangle(list(box), outline='red', width=2)
        if labels is not None:
            draw.text((box[0], box[1]), str(labels[i]), fill='red')
    plt.imshow(img)
    plt.axis('off')
    plt.show() 