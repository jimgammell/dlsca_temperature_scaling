# Implementation of the models proposed in 'Revisiting a Methodology for Efficient CNN Architectures in Profiling Attacks' by Wouters et al.
# Adapted from https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from models.common import VerboseModule

class FeatureExtractorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()

        self.conv = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2
            )
        self.act = nn.SELU()
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

class WoutersNet(VerboseModule):
    def __init__(
        self,
        input_shape,
        output_classes=256,
        input_pool_size=2,
        fe_channels=[64, 128],
        fe_kernel_sizes=[25, 3],
        fe_pool_sizes=[25, 4],
        classifier_layer_widths=[20, 20, 20]
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        
        # Ensure that inputs are valid
        if not len(fe_channels) == len(fe_kernel_sizes) == len(fe_pool_sizes):
            assert False
        num_fe_layers = len(fe_channels)
        num_classifier_layers = len(classifier_layer_widths)
        
        # Construct the feature extractor
        feature_extractor = nn.Sequential(OrderedDict(
            [('input_pool', nn.AvgPool1d(kernel_size=input_pool_size, stride=input_pool_size))] +
            [
                ('block_%d'%(layer_idx), FeatureExtractorBlock(in_channels, out_channels, kernel_size, pool_size))
                for layer_idx, in_channels, out_channels, kernel_size, pool_size in zip(
                    range(num_fe_layers), [input_shape[0]]+fe_channels[:-1], fe_channels, fe_kernel_sizes, fe_pool_sizes
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
                    ('act_%d'%(layer_idx), nn.SELU())
                 ] for layer_idx, in_activations, out_activations in zip(
                range(num_classifier_layers), [fe_activation_count]+classifier_layer_widths, classifier_layer_widths+[output_classes])], start=[]) +
            [('dense_%d'%(num_classifier_layers), nn.Linear(classifier_layer_widths[-1], output_classes))]
            
        ))
        
        # Construct the full model
        self.model = nn.Sequential(OrderedDict([
            ('feature_extractor', feature_extractor),
            ('flatten', nn.Flatten()),
            ('classifier', classifier)
        ]))
        
        self.apply(self.init_weights) # Initialize weights according to 'init_weights' function below
        
    def init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='selu') # a.k.a. the He uniform initialization
            module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
            
    def forward(self, x):
        return self.model(x)
        
class WoutersNet__ASCAD_Desync0(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[],
            fe_kernel_sizes=[],
            fe_pool_sizes=[],
            classifier_layer_widths=[10, 10],
            **kwargs
        )
        
class WoutersNet__ASCAD_Desync50(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[64, 128],
            fe_kernel_sizes=[25, 3],
            fe_pool_sizes=[25, 4],
            classifier_layer_widths=[20, 20, 20],
            **kwargs
        )
        
class WoutersNet__ASCAD_Desync100(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[64, 128],
            fe_kernel_sizes=[50, 3],
            fe_pool_sizes=[50, 2],
            classifier_layer_widths=[20, 20, 20],
            **kwargs
        )
        
class WoutersNet__DPAv4(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[],
            fe_kernel_sizes=[],
            fe_pool_sizes=[],
            classifier_layer_widths=[2],
            **kwargs
        )

class WoutersNet__AES_RD(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[16, 32],
            fe_kernel_sizes=[50, 3],
            fe_pool_sizes=[50, 7],
            classifier_layer_widths=[10, 10],
            **kwargs
        )
        
class WoutersNet__AES_HD(WoutersNet):
    def __init__(self, **kwargs):
        super().__init__(
            input_pool_size=2,
            fe_channels=[],
            fe_kernel_sizes=[],
            fe_pool_sizes=[],
            classifier_layer_widths=[2],
            **kwargs
        )

AVAILABLE_MODELS = [
    WoutersNet__ASCAD_Desync0, WoutersNet__ASCAD_Desync50, WoutersNet__ASCAD_Desync100,
    WoutersNet__AES_HD, WoutersNet__AES_RD, WoutersNet__DPAv4
]