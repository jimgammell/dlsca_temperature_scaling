# Adapted from https://github.com/rishikksh20/MLP-Mixer-pytorch/

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange

from models.common import VerboseModule

class Feedforward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0.0
    ):
        super().__init__()
        
        self.dropout = dropout
        
        self.dense_0 = nn.Linear(input_dim, hidden_dim)
        self.act_0 = nn.GELU()
        if dropout > 0:
            self.dropout_0 = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(hidden_dim, input_dim)
        if self.dropout > 0:
            self.dropout_1 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dense_0(x)
        x = self.act_0(x)
        if self.dropout > 0:
            x = self.dropout_0(x)
        x = self.dense_1(x)
        if self.dropout > 0:
            x = self.dropout_1(x)
        return x

class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches,
        in_dims,
        hidden_spatial_dims,
        hidden_channel_dims,
        dropout=0.0
    ):
        super().__init__()
        
        self.spatial_mixer = nn.Sequential(
            nn.LayerNorm(in_dims),
            Rearrange('b n d -> b d n'),
            Feedforward(num_patches, hidden_spatial_dims, dropout),
            Rearrange('b d n -> b n d')
        )
        
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(in_dims),
            Feedforward(in_dims, hidden_channel_dims, dropout=dropout)
        )
        
    def forward(self, x):
        x = x + self.spatial_mixer(x)
        x = x + self.channel_mixer(x)
        return x

class MLPMixer(VerboseModule):
    def __init__(
        self,
        input_shape,
        output_classes=256,
        patch_length=50, # How many samples each patch should correspond to
        dims=256, # Dimensions of each patch
        spatial_hidden_dims=256, # Hidden layer width for the spatial MLPs
        channel_hidden_dims=512, # Hidden layer width for the channel MLPs
        depth=8, # How many mixer blocks should be in the full model,
        dropout=0.0 # Applied after each layer in each of the multilayer perceptrons
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        
        num_patches = int(np.ceil(input_shape[1]/patch_length))
        input_padding = 0 if (input_shape[1] % patch_length) == 0 else patch_length - (input_shape[1] % patch_length)
        patch_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=input_shape[0], out_channels=dims, kernel_size=patch_length, stride=patch_length, padding=input_padding
            ),
            Rearrange('b c n -> b n c')
        )
        
        mixer_blocks = nn.Sequential(OrderedDict([
            ('block_%d'%(layer_idx), MixerBlock(num_patches, dims, spatial_hidden_dims, channel_hidden_dims, dropout=dropout))
            for layer_idx in range(depth)
        ]))
        
        global_average_pooling = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        classifier = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, output_classes)
        )
        
        self.model = nn.Sequential(OrderedDict([
            ('patch_embedding', patch_embedding),
            ('mixer_blocks', mixer_blocks),
            ('global_average_pooling', global_average_pooling),
            ('classifier', classifier)
        ]))
        
    def forward(self, x):
        return self.model(x)

AVAILABLE_MODELS = [
    MLPMixer
]