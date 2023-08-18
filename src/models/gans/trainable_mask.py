import numpy as np
import torch
from torch import nn

class FixedMask(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.mask = nn.Parameter(torch.zeros(*input_shape, dtype=torch.float))
        
    def forward(self, x):
        return self.mask.expand(*x.size())

class VGGClassifier(nn.Module):
    def __init__(self, input_shape, cnn_kernels=[16, 32, 64], layer_sizes=[64, 64], kernel_size=11):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(*sum([
            [nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm1d(c2), nn.ReLU(), nn.AvgPool1d(2)]
            for c1, c2 in zip([1]+cnn_kernels[:-1], cnn_kernels)
        ], start=[]))
        self.classifier = nn.Sequential(*sum([
            [nn.Linear(c1, c2), nn.ReLU()]
            for c1, c2 in zip([cnn_kernels[-1]]+layer_sizes[:-1], layer_sizes)
        ], start=[]), nn.Linear(layer_sizes[-1], 256))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return x

AVAILABLE_MODELS = [FixedMask, VGGClassifier]