# Implementation of the models proposed in 'Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database' by Prouff et al.
# Adapted from https://github.com/ANSSI-FR/ASCAD/

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from models.common import VerboseModule

class FeatureExtractorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2
        )
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class ProuffMLP(VerboseModule):
    def __init__(
        self,
        input_shape,
        output_classes=256
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        
        layer_width = 200
        num_layers = 6
        
        self.model = nn.Sequential(OrderedDict(
            [('flatten', nn.Flatten()),
             ('dense_0', nn.Linear(np.prod(input_shape), layer_width)),
             ('act_0', nn.ReLU(inplace=True))] +
            sum([
                [('dense_%d'%(layer_idx), nn.Linear(layer_width, layer_width)),
                 ('act_%d'%(layer_idx), nn.ReLU(inplace=True))]
                for layer_idx in range(1, num_layers-1)
            ], start=[]) +
            [('dense_%d'%(num_layers-1), nn.Linear(layer_width, output_classes))]
        ))
        
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu')) # a.k.a. the Glorot uniform initialization which Keras uses by default
            module.bias.data.zero_()
            
    def forward(self, x):
        return self.model(x)
             
class ProuffVGG(VerboseModule):
    def __init__(
        self,
        input_shape,
        output_classes=256,
        input_stride=1 # Stride setting for first conv layer. In the original work this was set to 1 for 700-sample input traces (fixed-key ASCAD) and to 2 for the 1400-sample input traces (variable-key ASCAD).
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        
        fe_channels = [min(64*2**n, 512) for n in range(5)]
        fe_kernel_sizes = [11 for _ in fe_channels]
        classifier_layer_widths = [4096, 4096]
        num_fe_layers = 5
        num_classifier_layers = 2
        
        # Construct the feature extractor
        feature_extractor = nn.Sequential(OrderedDict([
            ('block_%d'%(layer_idx), FeatureExtractorBlock(
                in_channels, out_channels, kernel_size, stride=input_stride if layer_idx==0 else 1
            ))
            for layer_idx, in_channels, out_channels, kernel_size in zip(
                range(num_fe_layers), [input_shape[0]]+fe_channels[:-1], fe_channels, fe_kernel_sizes
            )
        ]))
        
        # Calculate how many activations will be in the flattened feature extractor output
        eg_input = torch.randn(1, *input_shape)
        eg_fe_output = feature_extractor(eg_input)
        fe_activation_count = np.prod(eg_fe_output.shape)
        
        # Construct the multilayer perceptron classifier
        classifier = nn.Sequential(OrderedDict(
            sum([[
                    ('dense_%d'%(layer_idx), nn.Linear(in_activations, out_activations)),
                    ('act_%d'%(layer_idx), nn.ReLU(inplace=True))
                 ] for layer_idx, in_activations, out_activations in zip(
                range(num_classifier_layers), [fe_activation_count]+classifier_layer_widths, classifier_layer_widths+[output_classes]
                )], start=[]) +
            [('dense_%d'%(num_classifier_layers), nn.Linear(classifier_layer_widths[-1], output_classes))]
        ))
        
        # Construct the full model
        self.model = nn.Sequential(OrderedDict([
            ('feature_extractor', feature_extractor),
            ('flatten', nn.Flatten()),
            ('classifier', classifier)
        ]))
        
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu')) # a.k.a. the Glorot uniform initialization which Keras uses by default
            module.bias.data.zero_()
        
    def forward(self, x):
        return self.model(x)

AVAILABLE_MODELS = [
    ProuffMLP, ProuffVGG
]