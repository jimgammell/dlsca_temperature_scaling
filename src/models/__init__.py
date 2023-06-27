import os
import importlib
import torch
from torch import nn

AVAILABLE_MODELS = []
def add_models_from_module(mod):
    global AVAILABLE_MODELS
    for submod_file in os.listdir(os.path.join(os.path.dirname(__file__), mod)):
        if not submod_file.split('.')[-1] == 'py':
            continue
        submod = importlib.import_module('.'+submod_file.split('.')[0], 'models.'+mod)
        if not hasattr(submod, 'AVAILABLE_MODELS'):
            continue
        AVAILABLE_MODELS += submod.AVAILABLE_MODELS
add_models_from_module('classifiers')
add_models_from_module('gans')

def construct_model(model_name, **model_kwargs):
    model_classes = [m for m in AVAILABLE_MODELS if m.__name__ == model_name]
    assert len(model_classes) <= 1
    if len(model_classes) == 0:
        raise ValueError(
            'Invalid model name: {}. Choose from the following implemented models: [\n{}\n]'.format(
                model_name,
                ',\t\n'.join([m.__name__ for m in AVAILABLE_MODELS])
            ))
    model_class = model_classes[0]
    model = model_class(**model_kwargs)
    return model

def test(input_shape=(1, 700)):
    for model_class in AVAILABLE_MODELS:
        model = model_class(input_shape=input_shape)
        print(model)
        print()