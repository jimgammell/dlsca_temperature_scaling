import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

PreciseBatchNorm = type('PreciseBatchNorm', (_BatchNorm,), {'momentum': None, 'track_running_stats': True})

class VerboseModule(nn.Module):
    def save_input_shape(self, input_shape):
        self.input_shape = input_shape
        eg_input = torch.randn(1, *input_shape)
        modules = sum([list(mod.children()) if isinstance(mod, nn.Sequential) else [mod]
                      for mod in self.children()], start=[])
        for mod in modules:
            setattr(mod, 'input_shape', eg_input.shape[1:])
            eg_input = mod(eg_input)

    def extra_repr(self):
        info = []
        info.append('Parameter count: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        if hasattr(self, 'input_shape'):
            eg_input = torch.randn(
                1, *self.input_shape,
                device=next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
            )
            eg_output = self(eg_input)
            info.append('Input shape: {} -> Output shape: {}'.format(eg_input.shape[1:], eg_output.shape[1:]))
        return '\n'.join(info)