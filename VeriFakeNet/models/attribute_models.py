import torch
import torch.nn as nn
from torchvision import models

class GenderClassifier(nn.Module):
    """
    Predicts Male vs Female (Binary: 0=Male, 1=Female)
    """
    def __init__(self, pretrained=True):
        super(GenderClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)

class SkinToneClassifier(nn.Module):
    """
    Predicts Skin Tone / Demographic Group (White, Black, Asian, Indian, Other)
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(SkinToneClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class HairTextureClassifier(nn.Module):
    """
    Predicts Hair Texture: Straight, Wavy, Curly, Dreadlocks, Kinky
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(HairTextureClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class MultiAttributeFaceModel(nn.Module):
    """
    Combined Face Attributes Neural Network for fast joint inference.
    """
    def __init__(self, gender_classes=1, skin_classes=5, hair_classes=5, pretrained=True):
        super(MultiAttributeFaceModel, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        num_features = self.backbone.classifier[0].in_features
        
        self.backbone.classifier = nn.Identity()
        
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, gender_classes)
        )
        self.skin_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, skin_classes)
        )
        self.hair_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, hair_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        gender_out = self.gender_head(feat)
        skin_out = self.skin_head(feat)
        hair_out = self.hair_head(feat)
        return {
            'gender': gender_out,
            'skin_tone': skin_out,
            'hair_texture': hair_out
        }
