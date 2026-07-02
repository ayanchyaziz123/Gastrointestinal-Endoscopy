import torch.nn as nn
from torchvision import models
import timm

from .config import NUM_CLASSES, DROPOUT_RATES


def build_densenet121(num_classes=NUM_CLASSES):
    m = models.densenet121(weights='IMAGENET1K_V1')
    m.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATES['densenet121']),
        nn.Linear(m.classifier.in_features, num_classes),
    )
    return m


def build_efficientnet_b0(num_classes=NUM_CLASSES):
    m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    m.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATES['efficientnet_b0']),
        nn.Linear(m.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def build_deit_tiny(num_classes=NUM_CLASSES):
    m = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=0)
    m.head = nn.Sequential(
        nn.Dropout(DROPOUT_RATES['deit_tiny']),
        nn.Linear(m.num_features, num_classes),
    )
    return m
