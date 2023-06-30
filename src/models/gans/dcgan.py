import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class DCGAN__Generator__DONTUSE(nn.Module):
    def __init__(
        self,
        input_shape,
        base_channels=16,
        kernel_size=11,
        num_blocks=3,
        embedding_dim=256,
        use_sn=False,
        use_sample_scalar=False,
        use_gamma=False,
        skip_connection=False
    ):
        super().__init__()
        
        self.skip_connection = skip_connection
        
        def get_block(in_channels, out_channels, downsample=True, use_bn=True):
            modules = [] if downsample else [nn.Upsample(scale_factor=2)]
            modules.append(nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2 if downsample else 1
            ))
            if use_bn:
                modules.append(nn.BatchNorm1d(out_channels))
            modules.append(nn.ReLU(inplace=True))
            modules = nn.Sequential(*modules)
            return modules
        
        self.downsample_blocks = nn.ModuleList([
            get_block(input_shape[0], base_channels, downsample=True, use_bn=False),
            *[get_block(base_channels*2**n, base_channels*2**(n+1), downsample=True) for n in range(num_blocks-1)]
        ])
        self.upsample_blocks = nn.ModuleList([
            get_block(base_channels*2**(num_blocks-1), base_channels*2**(num_blocks-2), downsample=False),
            *[get_block(base_channels*2**(n+1), base_channels*2**(n-1), downsample=False) for n in range(num_blocks-2, 0, -1)],
            get_block(2*base_channels, input_shape[0], downsample=False, use_bn=False)
        ])
        if use_sample_scalar:
            self.sample_scalar = nn.Sequential(
                nn.Linear(base_channels*2**num_blocks + embedding_dim, base_channels*2**num_blocks),
                nn.ReLU(inplace=True),
                nn.Linear(base_channels*2**num_blocks, input_shape[1]),
                nn.Sigmoid()
            )
        if use_gamma:
            self.register_parameter('gamma', torch.tensor(0, dtype=torch.float))
        
        if use_sn:
            for mod in self.modules():
                if isinstance(mod, (nn.Conv1d, nn.Linear)):
                    mod = spectral_norm(mod)
        self.apply(self.init_weights_)
        
    def init_weights_(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        
    def forward(self, x):
        if self.skip_connection:
            x_orig = x
        x_i = self.downsample_blocks[0](x)
        x_skip = [x_i]
        for ds_block in self.downsample_blocks[1:]:
            x_skip.append(ds_block(x_skip[-1]))
        x_resid = self.upsample_blocks[0](x_skip[-1])
        for us_block, x_s in zip(self.upsample_blocks[1:], x_skip[-2::-1]):
            x_resid = us_block(torch.cat([x_resid, x_s], dim=1))
        x_resid = nn.functional.tanh(x_resid)
        if hasattr(self, 'sample_scalar'):
            intermediate_features = x_skip[-1].mean(dim=-1)
            sample_scalar = self.sample_scalar(torch.cat([intermediate_features, y_embedded], dim=1))
            x_resid = x_resid * sample_scalar
        if hasattr(self, 'gamma'):
            x_resid = self.gamma*x_resid
        if self.skip_connection:
            out = x_orig + x_resid
        else:
            out = x_resid
        return out

class DCGAN__Generator(nn.Module):
    def __init__(
        self,
        input_shape,
        base_channels=32,
        kernel_size=11,
        num_blocks=3
    ):
        super().__init__()
        
        def get_block(in_channels, out_channels, use_bn=True):
            modules = [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True)
            ]
            if use_bn:
                modules.append(nn.BatchNorm1d(out_channels))
            return nn.Sequential(*modules)
        
        self.input_transform = nn.Sequential(
            nn.Conv1d(input_shape[0], base_channels, kernel_size=kernel_size, padding=kernel_size//2)
        )
        self.trunk = nn.Sequential(*[
            get_block(base_channels, base_channels) for _ in range(num_blocks)
        ])
        self.output_transform = nn.Sequential(
            nn.Conv1d(base_channels, input_shape[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.Tanh()
        )
        self.output_scalar = nn.Parameter(torch.tensor(0, dtype=torch.float))
        
    def forward(self, x):
        #x_orig = x.clone()
        x = self.input_transform(x)
        x = self.trunk(x)
        x = self.output_transform(x)
        x = self.output_scalar*x
        return x
    
class DCGAN__Discriminator(nn.Module):
    def __init__(
        self,
        input_shape,
        base_channels=16,
        kernel_size=11,
        num_blocks=3,
        use_sn=False
    ):
        super().__init__()
        
        def get_block(in_channels, out_channels, use_bn=True):
            modules = [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
                nn.LeakyReLU(0.2)
            ]
            if use_bn:
                modules.append(nn.BatchNorm1d(out_channels))
            return nn.Sequential(*modules)
        
        self.feature_extractor = nn.Sequential(*(
            [get_block(input_shape[0], base_channels, use_bn=False)] + sum(
                [[get_block(base_channels*2**n, base_channels*2**(n+1))] for n in range(num_blocks-1)], start=[]
            ) + [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        ))
        num_features = base_channels*2**(num_blocks-1)
        self.head = nn.Linear(num_features, 256)
        
        if use_sn:
            for mod in self.modules():
                if isinstance(mod, (nn.Linear, nn.Conv1d)):
                    mod = spectral_norm(mod)
        self.apply(self.init_weights_)
        
    def init_weights_(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
            module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
            
    def forward(self, x):
        x_fe = self.feature_extractor(x)
        out = self.head(x_fe)
        return out

AVAILABLE_MODELS = [DCGAN__Generator, DCGAN__Discriminator]