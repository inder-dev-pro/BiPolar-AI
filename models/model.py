import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes):
    # Load Faster R-CNN pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (for Pascal VOC)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model 