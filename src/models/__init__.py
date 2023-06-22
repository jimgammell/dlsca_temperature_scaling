import os
import importlib
import torch
from torch import nn

AVAILABLE_MODELS = []
for mod_name in os.listdir(os.path.join(os.path.dirname(__file__), 'classifiers')):
    if not mod_name.split('.')[-1] == 'py':
        continue
    mod = importlib.import_module('.'+mod_name.split('.')[0], 'models.classifiers')
    if not hasattr(mod, 'AVAILABLE_MODELS'):
        continue
    AVAILABLE_MODELS += mod.AVAILABLE_MODELS

def construct_model(model_name, **model_kwargs):
    model_classes = [m for m in AVAILABLE_MODELS if m.__name__ == model_name]
    assert len(model_classes) <= 1
    if len(model_classes) == 0:
        raise ValueError(
            'Invalid model name: {}. Choose from the following implemented models: [\n{}\n]'.format(
                model_name,
                ',\t\n'.join([m.__name__ for m in AVAILABLE_MODELS])
            ))
    model = model_classes[0](**model_kwargs)
    return model

def test(input_shape=(1, 700)):
    for model_class in AVAILABLE_MODELS:
        model = model_class(input_shape=input_shape)
        print(model)
        print()