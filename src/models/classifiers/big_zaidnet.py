from copy import copy
import numpy as np
import torch
from torch import nn

from models.common import VerboseModule

# Source: https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_type = pad_type
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size-1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        
        a = [1.] # Pascal's triangle
        for _ in range(filt_size-1):
            a = [1., *[e1 + e2 for e1, e2 in zip(a[:-1], a[1:])], 1.]
        a = np.array(a)
        filt = torch.tensor(a, dtype=torch.float)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt.view(1, 1, -1).repeat((self.channels, 1, 1)))
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)
        
    def forward(self, x):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return x[:, :, ::self.stride]
            else:
                return self.pad(x)[:, :, ::self.stride]
        else:
            return nn.functional.conv1d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])
    
    def __repr__(self):
        return self.__class__.__name__ + '(channels={}, filter_size={}, stride={}, pad_type={}, pad_offset={})'.format(
            self.channels, self.filt_size, self.stride, self.pad_type, self.pad_off
        )

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

 # To try: increase conv stride, reduce antialiasing stride
class FEBlock(VerboseModule):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_method='aa_sconv'):
        super().__init__()
        
        if pooling_method == 'aa_sconv':
            self.conv = nn.Sequential(
                BlurPool(in_channels, filt_size=kernel_size, stride=5),
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            )
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.selu = nn.SELU()
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.9)
        if pooling_method == 'aa_maxpool':
            self.pool = nn.Sequential(
                nn.MaxPool1d(5, stride=1),
                BlurPool(out_channels, filt_size=kernel_size, stride=5)
            )
        elif pooling_method == 'aa':
            self.pool = BlurPool(out_channels, filt_size=kernel_size, stride=5)
        elif pooling_method == 'avgpool':
            self.pool = nn.AvgPool1d(5)
        elif pooling_method == 'none':
            self.pool = nn.Identity()
        else:
            assert False
        
    def forward(self, x):
        x = self.conv(x)
        x = self.selu(x)
        x = self.bn(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x

class MLPBlock(VerboseModule):
    def __init__(self, in_dims, out_dims, use_act=True):
        super().__init__()
        
        self.dense = nn.Linear(in_dims, out_dims)
        if use_act:
            self.selu = nn.SELU()
    
    def forward(self, x):
        x = self.dense(x)
        if hasattr(self, 'selu'):
            x = self.selu(x)
        return x

class BigZaidNet(VerboseModule):
    def __init__(
        self,
        input_shape,
        fe_blocks=3,
        mlp_blocks=3,
        mlp_dims=256,
        pooling_method='avgpool'
    ):
        super().__init__()
        
        initial_channels = mlp_dims // 2**(fe_blocks-1)
        self.feature_extractor = nn.Sequential(
            FEBlock(input_shape[0], initial_channels, 11, pooling_method='none'),
            *[FEBlock(initial_channels*2**n, initial_channels*2**(n+1), 11, pooling_method=pooling_method)
              for n in range(fe_blocks-1)]
        )
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.multilayer_perceptron = nn.Sequential(
            *[MLPBlock(mlp_dims, mlp_dims) for _ in range(mlp_blocks-1)],
            MLPBlock(mlp_dims, 256, use_act=False)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_average_pooling(x)
        x = self.multilayer_perceptron(x)
        return x

AVAILABLE_MODELS = [BigZaidNet]