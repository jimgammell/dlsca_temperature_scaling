import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class AdaptiveInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, num_features, embedding_dim=64, *args, **kwargs):
        assert embedding_dim is not None
        super().__init__(num_features, *args, affine=False, **kwargs)
        self.get_affine = nn.Linear(embedding_dim, 2*num_features)
        
    def forward(self, x, y):
        batch_size, channels, _ = x.size()
        affine_params = self.get_affine(y)
        gamma, beta = torch.split(affine_params, channels, dim=1)
        gamma, beta = gamma.view(batch_size, channels, 1), beta.view(batch_size, channels, 1)
        x_norm = super().forward(x)
        scalar = 1 + gamma
        out = scalar * x_norm + beta
        return out

class Block(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=7,
        activation=None,
        resample_layer=None,
        norm_layer=None
    ):
        super().__init__()
        
        self.resid_modules = nn.ModuleList([])
        if norm_layer is not None:
            self.resid_modules.append(norm_layer(in_channels))
        if activation is not None:
            self.resid_modules.append(activation())
        self.resid_modules.append(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2))
        if resample_layer == 'downsample':
            self.resid_modules.append(nn.AvgPool1d(2))
        elif resample_layer == 'upsample':
            self.resid_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif resample_layer == 'none':
            pass
        else:
            assert False
        if norm_layer is not None:
            self.resid_modules.append(norm_layer(in_channels))
        if activation is not None:
            self.resid_modules.append(activation())
        self.resid_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        
        self.skip_modules = nn.ModuleList([])
        if in_channels != out_channels:
            self.skip_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        if resample_layer == 'downsample':
            self.skip_modules.append(nn.AvgPool1d(2))
        elif resample_layer == 'upsample':
            self.skip_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        
    def forward(self, *args):
        if len(args) == 1:
            (x,) = args
            y = None
        elif len(args) == 2:
            (x, y) = args
        else:
            assert False
        
        x_resid = x.clone()
        for resid_mod in self.resid_modules:
            if isinstance(resid_mod, AdaptiveInstanceNorm1d):
                assert y is not None
                x_resid = resid_mod(x_resid, y)
            else:
                x_resid = resid_mod(x_resid)
        
        x_skip = x
        for skip_mod in self.skip_modules:
            x_skip = skip_mod(x_skip)
        
        out = (x_resid + x_skip) / np.sqrt(2)
        return out

class StarGAN__Generator(nn.Module):
    def __init__(
        self,
        input_shape,
        residual_model=False,
        base_channels=16,
        resample_blocks=2,
        isotropic_blocks=0,
        kernel_size=3,
        embedding_dim=64,
        min_perturbation_thresh=1e-3,
        use_sn=True
    ):
        super().__init__()
        
        self.residual_model = residual_model
        self.min_perturbation_thresh = min_perturbation_thresh
        
        activation = lambda: nn.ReLU(inplace=True)
        ds_norm_layer = lambda num_features: nn.InstanceNorm1d(num_features, affine=True)
        us_norm_layer = lambda num_features: AdaptiveInstanceNorm1d(num_features, embedding_dim=embedding_dim)
        
        self.input_transform = nn.Conv1d(input_shape[0], base_channels, kernel_size=1)
        self.downsample_blocks = nn.ModuleList([
            Block(
                base_channels*2**n, base_channels*2**(n+1), kernel_size=kernel_size,
                activation=activation, resample_layer='downsample', norm_layer=ds_norm_layer
            ) for n in range(resample_blocks)
        ])
        self.isotropic_blocks = nn.ModuleList([
            Block(
                base_channels*2**resample_blocks, base_channels*2**resample_blocks, kernel_size=kernel_size,
                activation=activation, resample_layer='none', norm_layer=ds_norm_layer
            ) for _ in range(isotropic_blocks//2)] + [
            Block(
                base_channels*2**resample_blocks, base_channels*2**resample_blocks, kernel_size=kernel_size,
                activation=activation, resample_layer='none', norm_layer=us_norm_layer
            ) for _ in range(isotropic_blocks//2 + isotropic_blocks%2)
        ])
        self.upsample_blocks = nn.ModuleList([
            Block(
                base_channels*2**n, base_channels*2**(n-1), kernel_size=kernel_size,
                activation=activation, resample_layer='upsample', norm_layer=us_norm_layer
            ) for n in range(resample_blocks, 0, -1)
        ])
        self.output_transform = nn.Conv1d(base_channels, input_shape[0], kernel_size=1)
        self.class_embedding = nn.Linear(256, embedding_dim)
        self.recombine_blocks = nn.ModuleList([
            nn.Conv1d(base_channels*2**(n+1), base_channels*2**n, kernel_size=1)
            for n in range(resample_blocks+1)
        ][::-1])
        
        if use_sn:
            for mod in self.modules():
                if isinstance(mod, (nn.Conv1d, nn.Linear)):
                    mod = spectral_norm(mod)
        
    def forward(self, x, y):
        if self.residual_model:
            x_orig = x.clone()
        if y.ndim < 2:
            y = nn.functional.one_hot(y, num_classes=256).to(torch.float)
        y_embedded = self.class_embedding(y)
        x = self.input_transform(x)
        x_scales = [x.clone()]
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
            x_scales.append(x.clone())
        x_scales = x_scales[::-1]
        for iso_block in self.isotropic_blocks:
            x = iso_block(x, y_embedded)
        x = torch.cat((x, x_scales[0]), dim=1)
        x = self.recombine_blocks[0](x)
        for us_block, x_scale, recomb_block in zip(self.upsample_blocks, x_scales[1:], self.recombine_blocks[1:]):
            x = us_block(x, y_embedded)
            if x.size(2) < x_scale.size(2):
                pad_amt = x_scale.size(2) - x.size(2)
                x = nn.functional.pad(x, (pad_amt//2, pad_amt//2+pad_amt%2))
            x = torch.cat((x, x_scale), dim=1)
            x = recomb_block(x)
        x = self.output_transform(x)
        x = nn.functional.tanh(x)
        if self.residual_model:
            if not self.training and self.min_perturbation_thresh > 0:
                # When in evaluation mode, prune perturbation values which don't exceed a certain magnitude
                x = torch.where(x.abs() > self.min_perturbation_thresh, x, torch.zeros_like(x))
            x = x + x_orig
        return x

class StarGAN__Discriminator(nn.Module):
    def __init__(
        self,
        input_shape,
        base_channels=16,
        ds_blocks=2,
        iso_blocks=0,
        kernel_size=11,
        embedding_dim=64,
        use_sn=True
    ):
        super().__init__()
        
        activation = lambda: nn.LeakyReLU(negative_slope=0.1)
        norm_layer = lambda num_features: nn.InstanceNorm1d(num_features, affine=True)
        
        self.input_transform = nn.Conv1d(input_shape[0], base_channels, kernel_size=1)
        self.downsample_blocks = nn.Sequential(*[
            Block(
                base_channels*2**n, base_channels*2**(n+1), kernel_size=kernel_size,
                activation=activation, resample_layer='downsample', norm_layer=norm_layer
            ) for n in range(ds_blocks)
        ])
        self.isotropic_blocks = nn.Sequential(*[
            Block(
                base_channels*2**ds_blocks, base_channels*2**ds_blocks, kernel_size=kernel_size,
                activation=activation, resample_layer='none', norm_layer=norm_layer
            ) for n in range(iso_blocks)
        ])
        eg_input = torch.randn(1, *input_shape)
        eg_input = self.input_transform(eg_input)
        eg_input = self.downsample_blocks(eg_input)
        eg_input = self.isotropic_blocks(eg_input)
        self.output_transform = nn.Sequential(
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier_heads = nn.ModuleDict({
            'realism': nn.Linear(base_channels*2**ds_blocks, 1),
            'naive_leakage': nn.Linear(base_channels*2**ds_blocks, 256),
            'adversarial_leakage': nn.Linear(base_channels*2**ds_blocks, 256)
        })
        self.class_embedding = nn.Linear(256, embedding_dim)
        
        if use_sn:
            for mod in self.modules():
                if isinstance(mod, (nn.Linear, nn.Conv1d)):
                    mod = spectral_norm(mod)
    
    def extract_features(self, x):
        x = self.input_transform(x)
        x = self.downsample_blocks(x)
        x = self.isotropic_blocks(x)
        x = self.output_transform(x)
        x = x.view(x.size(0), -1)
        return x
    
    def classify_realism(self, x, y):
        if y.ndim < 2:
            y = nn.functional.one_hot(y, 256).to(torch.float)
        embedded_y = self.class_embedding(y)
        x = self.classifier_heads['realism'](x) + (x*embedded_y).sum(dim=-1, keepdim=True)
        return x
    
    def classify_naive_leakage(self, x):
        x = self.classifier_heads['naive_leakage'](x)
        return x
    
    def classify_adversarial_leakage(self, x):
        x = self.classifier_heads['adversarial_leakage'](x)
        return x

AVAILABLE_MODELS = [StarGAN__Generator, StarGAN__Discriminator]