import torch
from torchvision import models

def load_model():
    """
    Loads the ResNet50 model for image classification.
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
