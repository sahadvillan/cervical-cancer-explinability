import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CervicalCancerModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CervicalCancerModel, self).__init__()
        # Load a pre-trained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer for our specific number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_target_layer(self):
        """
        Returns the target layer for Grad-CAM.
        For ResNet, this is usually the last convolutional layer in the last block (layer4[-1]).
        """
        return self.model.layer4[-1]
