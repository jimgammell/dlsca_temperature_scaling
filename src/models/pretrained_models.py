import copy
import numpy as np
import torch
from torch import nn

import resources

class PretrainedKerasModel(nn.Module):
    def __init__(self, model_path, device):
        import tensorflow as tf
        from tensorflow import keras
        
        super().__init__()
        
        # convert device identifier from PyTorch to TensorFlow format
        if device == 'cpu':
            self.tf_device = 'CPU'
        elif 'cuda:' in device:
            self.tf_device = 'GPU:%d'%(int(device.split(':')[-1]))
        else:
            assert False
        
        self.model = keras.models.load_model(model_path)
        # ^ remove implicit softmax at output; we will apply explicitly where appropriate
        self.model.layers[-1].activation = keras.activations.linear
        
    def forward(self, x):
        if isinstance(x, torch.Tensor): # TensorFlow likes numpy arrays as input
            x = x.cpu().numpy()
        x = x.transpose(0, 2, 1) # TensorFlow has channel as last dimension, whereas PyTorch has it first
        with tf.device(self.tf_device): # Do forward pass on appropriate device
            logits = self.model(x, training=False)
        logits = torch.tensor(np.array(logits), dtype=torch.float) # Convert output to what we would expect if this were a PyTorch model
        return logits
    
    def __repr__(self):
        # Return the Keras model summary as a string
        stringlist = [self.__class__.__name__+'(']
        self.model.summary(print_fn=lambda x: stringlist.append('\t'+x))
        stringlist.append(')')
        return '\n'.join(stringlist)

class PretrainedProuff(PretrainedKerasModel):
    def __init__(self, variable=False, desync=0, arch='cnn', device='cpu'):
        if variable:
            assert arch == 'cnn'
            from resources import ascadv1_variable
            assert desync in ascadv1_variable.VALID_DESYNC
            model_path = ascadv1_variable.get_model_path(desync)
        else:
            from resources import ascadv1_fixed
            assert desync in ascadv1_fixed.VALID_DESYNC
            assert arch in ascadv1_fixed.VALID_ARCH
            model_path = ascadv1_fixed.get_model_path(desync, arch)
        self.name = 'Prouff%s'%('CNN' if arch=='cnn' else 'MLP')
        super().__init__(model_path, device)

class PretrainedZaid(PretrainedKerasModel):
    def __init__(self, dataset='AES_HD', device='cpu'):
        from resources import zaid
        assert dataset in zaid.VALID_MODELS
        model_path = zaid.get_model_path(dataset)
        self.name = 'ZaidNet'
        super().__init__(model_path, device)