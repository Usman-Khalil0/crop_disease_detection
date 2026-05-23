import torch
import torch.nn as nn
import torchvision.models as models

# Define classes directly - no external config dependency
CLASS_NAMES = [
    'Healthy',
    'Early_blight',
    'Late_blight',
    'Leaf_mold',
    'Bacterial_spot',
    'Yellow_leaf_curl_virus',
    'Mosaic_virus'
]

NUM_CLASSES = len(CLASS_NAMES)  # 7 classes
MODEL_NAME = 'efficientnet_b3'

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, model_name=MODEL_NAME):
        super(DiseaseClassifier, self).__init__()
        
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, return_logits=False):
        logits = self.backbone(x)
        if return_logits:
            return logits
        return self.softmax(logits)


def get_model(num_classes=NUM_CLASSES, device='cpu'):
    model = DiseaseClassifier(num_classes=num_classes)
    model = model.to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Trainable parameters: {count_parameters(model):,}")