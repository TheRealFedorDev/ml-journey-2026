import torchvision.models as models
import torch

def create_model(num_classes, model_name='ResNet18'):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    num_classes = 4  # COVID, Normal, Lung Opacity, Viral Pneumonia
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model

