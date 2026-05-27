import torch
import torch.nn as nn
import torch.nn.functional as F

nc = 3    # number of color channels
nf = 64   # size of feature maps in generator
NUM_CLASSES = 10  # number of output classes for classification

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Bug 1: nn.sequential → nn.Sequential (capital S)
        self.features = nn.Sequential(
            # 3, 256, 256
            nn.Conv2d(nc, nf, 4, stride = 2, padding=1),    nn.BatchNorm2d(nf),     nn.ReLU(),
            # nf, 128, 128
            nn.Conv2d(nf, nf*2, 4, stride = 2, padding=1),  nn.BatchNorm2d(nf*2),   nn.ReLU(),
            # nf*2, 64, 64
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # ← the missing step
            nn.Linear(nf*2 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),           # raw logits, no Tanh
        )

    def forward(self, x):
        return self.layers(x)
