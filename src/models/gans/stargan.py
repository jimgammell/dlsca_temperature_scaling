import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class InstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, *args, use_sn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sn = use_sn
        
    def forward(self, *args, **kwargs):
        if self.weight is not None and self.use_sn:
            self.weight.data = torch.clip(self.weight.data, -1, 1)
        return super().forward(*args, **kwargs)
    
class AdaptiveInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, embedding_dim, num_features, *args, use_sn=False, **kwargs):
        assert embedding_dim is not None
        super().__init__(num_features, *args, affine=False, **kwargs)
        self.get_affine = nn.Linear(embedding_dim, 2*num_features)
        self.use_sn = use_sn
        
    def forward(self, x, y):
        batch_size, channels, _ = x.size()
        affine_params = self.get_affine(y)
        gamma, beta = torch.split(affine_params, channels, dim=-1)
        gamma, beta = gamma.view(batch_size, channels, 1), beta.view(batch_size, channels, 1)
        x_norm = super().forward(x)
        if self.use_sn:
            scalar = torch.clip(1 + gamma, -1, 1)
        else:
            scalar = 1 + gamma
        out = scalar * x_norm + beta
        return out

class Block(nn.Module):
    def __init__(
        self,
        channels, # Input channels to block. Will be multiplied by 2 if downsampling, or divided by 2 if upsampling.
        activation=type('LeakyReLU', (nn.LeakyReLU,), {'negative_slope': 0.1}),
        norm=type('InstanceNorm', (InstanceNorm1d,), {}),
        resample='none', # Whether to resample in this block. Valid options: ['none', 'downsample', 'upsample'].
    ):
        super().__init__()
        
        resid_path_modules = []
        if norm is not None:
            resid_path_modules.append(norm(channels))
        if activation is not None:
            resid_path_modules.append(activation())
        resid_path_modules.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2))
        if resample_layer == 'downsample':
            resid_path_modules.append(nn.AvgPool1d(2))
            out_channels = 2*channels
        elif resample_layer == 'upsample':
            resid_path_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
            out_channels = channels//2
        elif resample_layer == 'none':
            out_channels = channels
        else:
            assert False
        if norm is not None:
            resid_path_modules.append(norm(channels))
        if activation is not None:
            resid_path_modules.append(activation())
        resid_path_modules.append(nn.Conv1d(channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        
        skip_path_modules = []
        if channels != out_channels:
            skip_path_modules.append(nn.Conv1d(channels, out_channels, kernel_size=1, bias=False))
        if resample_layer == 'downsample':
            skip_path_modules.append(nn.AvgPool1d(2))
        elif resample_layer == 'upsample':
            skip_path_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.resid_path = nn.Sequential(*resid_path_modules)
        self.skip_path = nn.Sequential(*skip_path_modules)
        
    def forward(self, x):
        return self.resid_path(x) + self.skip_path(x)
        
class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape,
        head_sizes,
        base_channels=16,
        num_blocks=3,
        use_sn=True
    ):
        super().__init__()
        
        block_kwargs = {
            'activation': type('LeakyReLU', (nn.LeakyReLU,), {'negative_slope': 0.1}),
            'norm': None,
            'resample': 'downsample'
        }
        self.stem = nn.Conv1d(input_shape[0], base_channels, kernel_size=1, bias=False)
        self.trunk = nn.Sequential(*([
            Block(base_channels*2**n, **block_kwargs)
            for n in range(num_blocks)
        ] + [nn.AdaptiveAvgPool1d(1)]))
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base_channels*2**n, head_size, kernel_size=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(head_size, head_size, kernel_size=1),
                nn.Flatten()
            ) for head_size in head_sizes
        ])
        
        if use_sn:
            for mod in self.modules():
                if isinstance(mod, nn.Conv1d):
                    mod = spectral_norm(mod)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = [head(x) for head in self.heads]
        return x
