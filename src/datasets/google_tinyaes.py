import os
import time
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset

import resources
from datasets.common import DatasetBase

class GoogleTinyAES(DatasetBase):
    def __init__(
        self,
        train=True,
        resource_path=None,
        target_byte=0,
        interval_to_use=[0, 20000],
        target_attack_point='sub_bytes_in',
        **kwargs
    ):
        self.interval_to_use = interval_to_use
        self.target_byte = target_byte
        self.target_attack_point = target_attack_point
        
        if resource_path is None:
            from resources import google_scaaml
            resource_path = google_scaaml.get_dataset_path(train)
        self.resource_path = resource_path
        database_file = h5py.File(os.path.join(resource_path, 'data.hdf5'))
        self.data = database_file['traces']
        self.targets = database_file['{}__{}'.format(target_attack_point, target_byte)]
        self.metadata = {key: val for key, val in database_file.items() if key != 'traces'}
        self.length = len(self.data)
        assert self.length == len(self.targets)
        assert all(self.length == len(val) for val in self.metadata.values())
        self.name = 'Google_TinyAES'
        self.data_mean = -0.2751
        self.data_stdev = 0.1296
        
        super().__init__(train=train, **kwargs)
    
    def load_idx(self, idx):
        trace = self.data[idx, :, self.interval_to_use[0]:self.interval_to_use[1]]
        target = self.targets[idx]
        if hasattr(self, 'return_metadata') and self.return_metadata:
            metadata = {key: val[idx] for val in self.metadata.items()}
            return trace, target, metadata
        else:
            return trace, target

AVAILABLE_DATASETS = [GoogleTinyAES]