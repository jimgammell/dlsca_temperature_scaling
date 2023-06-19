import os
import numpy as np

import resources
from datasets.common import DatasetBase

class DPAv4_Zaid(DatasetBase):
    def __init__(self, train=True, resource_path=None, **kwargs):
        if resource_path is None:
            from resources import zaid
            resource_path = get_dataset_path()
        if train:
            self.data = np.load(os.path.join(resource_path, 'profiling_traces.npy')).astype(float)
            self.targets = np.load(os.path.join(resource_path, 'profiling_labels.npy')).astype(np.uint8)
            self.metadata = {
                'plaintext': np.load(os.path.join(resource_path, 'profiling_plaintext.npy')).astype(np.uint8)
            }
        else:
            self.data = np.load(os.path.join(resource_path, 'attack_traces.npy')).astype(float)
            self.targets = np.load(os.path.join(resource_path, 'attack_labels.npy')).astype(np.uint8)
            self.metadata = {
                'plaintext': np.load(os.path.join(resource_path, 'attack_plaintext.npy')).astype(np.uint8)
            }
        key = np.load(os.path.join(resource_path, 'key.npy')).astype(np.uint8)
        mask = np.load(os.path.join(resource_path, 'mask.npy')).astype(np.uint8)
        self.metadata['key'] = np.concatenate([key for _ in range(len(self.data))], axis=0)
        self.metadata['mask'] = np.concatenate([mask for _ in range(len(self.data))], axis=0)
        self.name = 'DPAv4_ZaidSubset'
        
        super().__init__(train=train, **kwargs)

AVAILABLE_DATASETS = [DPAv4_Zaid]