import torch
from torch import nn, optim

def decorate_model(
    model,
    dist_fn=lambda x1, x2: (x1-x2).norm(p=2),
    avg_fn=lambda x, x_avg: 0.9999*x_avg + 0.0001*x
):
    class AveragedModel(model.__class__):
        def init_avg(self):
            self.dist_fn = dist_fn
            self.avg_fn = avg_fn
            for param_name, param in self.named_params_grad():
                self.register_buffer(self.get_avg_param_name(param_name), param.data.clone())

        def get_avg_param_name(self, param_name):
            return 'avg__'+'_'.join(param_name.split('.'))

        def get_avg_param(self, name):
            return getattr(self, self.get_avg_param_name(name))

        def set_avg_param(self, name, val):
            setattr(self, self.get_avg_param_name(name), val.data.clone())

        def named_params_grad(self):
            return [(param_name, param) for param_name, param in self.named_parameters() if param.requires_grad]

        @torch.no_grad()
        def reset_avg_buffer(self):
            for param_name, param in self.named_params_grad():
                self.set_avg_param(param_name, param)

        @torch.no_grad()
        def update_avg_buffer(self):
            for param_name, param in self.named_params_grad():
                self.set_avg_param(param_name, self.avg_fn(param, self.get_avg_param(param_name)))

        def get_parameter_drift_loss(self):
            penalty = torch.sum(torch.stack([
                self.dist_fn(param, self.get_avg_param(param_name)) for param_name, param in self.named_params_grad()
            ]))
            return penalty
        
        def extra_repr(self):
            return super().extra_repr() + '\nRecording average parameters.'
    
    AveragedModel.__name__ = model.__class__.__name__
    model.__class__ = AveragedModel
    model.init_avg()