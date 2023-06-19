import numpy as np
import torch

from datasets import ascadv1, zaid_dpav4

AVAILABLE_DATASETS = sum([
    ascadv1.AVAILABLE_DATASETS, zaid_dpav4.AVAILABLE_DATASETS
], start=[])

def construct_dataset(dataset_name, **dataset_kwargs):
    dataset_classes = [d for d in AVAILABLE_DATASETS if d.__name__ == dataset_name]
    assert len(dataset_classes) <= 1
    if len(dataset_classes) == 0:
        raise ValueError(
            'Invalid dataset name: {}. Choose from the following implemented datasets: [\n{}\n]'.format(
                dataset_name, ',\t\n'.join([d.__name__ for d in AVAILABLE_DATASETS])
            ))
    dataset = dataset_classes[0](**dataset_kwargs)
    return dataset