# Object Detection Model Building Assignment

This repository contains a complete solution for an object detection assignment using a CNN backbone and Faster R-CNN. The codebase is modular, reproducible, and easy to extend.

## Directory Structure

```
object_detection_assignment/
│
├── data/                  # Dataset or download scripts
├── models/                # Model definitions (backbone, detection head)
├── notebooks/             # Jupyter notebooks for EDA or reporting
├── outputs/               # Trained weights, logs, results
├── src/                   # Source code (training, evaluation, inference, utils)
├── experience_report.md   # Experience report template
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Pascal VOC dataset (instructions in `data/` or via script)

## Usage

- **Training:**
  ```bash
  python src/train.py
  ```
- **Evaluation:**
  ```bash
  python src/eval.py
  ```
- **Inference/Demo:**
  ```bash
  python src/infer.py --image_path path/to/image.jpg
  ```

## Notes
- The code uses PyTorch and torchvision.
- The default backbone is ResNet-50 with Faster R-CNN.
- The dataset is Pascal VOC 2007/2012.
- See `experience_report.md` for the report template. 