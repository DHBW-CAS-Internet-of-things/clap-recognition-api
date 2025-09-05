import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Small CNN for audio classification using PyTorch.
    """

    def __init__(self, n_mels=40, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, time]
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.classifier(z)
