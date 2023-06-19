import torch
from torch import nn

PreciseBatchNorm = type('PreciseBatchNorm', (nn._BatchNorm,), {'momentum': None, 'track_running_stats': True})

class VerboseModule(nn.Module):
    def extra_repr(self):
        info = []
        info.append('Parameter count: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        if hasattr(self, 'input_shape'):
            eg_input = torch.randn(1, *self.input_shape)
            eg_output = self(eg_input)
            info.append('Input shape: {} -> Output shape: {}'.format(eg_input.shape[1:], eg_output.shape[1:]))
        return '\n'.join(info)
