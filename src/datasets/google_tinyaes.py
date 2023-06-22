import os
import numpy as np
from tqdm import tqdm

import resources
from datasets.common import DatasetBase

class GoogleTinyAES(DatasetBase):
    def __init__(
        self,
        train=True,
        resource_path=None,
        store_in_memory=True,
        target_byte=0,
        interval_to_use=[0, 20000],
        downsample_ratio=4,
        target_attack_point='sub_bytes_in',
        **kwargs
    ):
        self.store_in_memory = store_in_memory
        self.interval_to_use = interval_to_use
        self.target_byte = target_byte
        self.downsample_ratio = downsample_ratio
        self.target_attack_point = target_attack_point
        
        if resource_path is None:
            from resources import google_scaaml
            resource_path = google_scaaml.get_dataset_path(train)
        self.resource_path = resource_path
        if store_in_memory:
            self.data, self.targets, self.metadata = [], [], {}
            print('Loading data into memory ...')
            for filename in tqdm(os.listdir(resource_path)):
                traces, target, metadata = self.load_shard(filename)
                self.data.append(traces)
                self.targets.append(target)
                for key in self.metadata.keys():
                    if not key in self.metadata.keys():
                        self.metadata[key] = []
                    self.metadata[key].append(metadata[key])
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
            self.metadata = {key: np.concatenate(val, axis=0) for key, val in self.metadata.items()}
        else:
            self.filenames = [f for f in os.listdir(resource_path)]
            self.file_lengths = np.cumsum([0] + [len(np.load(os.path.join(resource_path, f))['traces']) for f in self.filenames])
            self.length = self.file_lengths[-1]
        
        self.name = 'Google_TinyAES'
        self.data_mean = -0.2751
        self.data_stdev = 0.1296
        
        super().__init__(train=train, **kwargs)
    
    def load_shard(self, filename, idx=None):
        shard = np.load(os.path.join(self.resource_path, filename))
        traces = shard['traces'][:, self.interval_to_use[0]:self.interval_to_use[1], :]
        target = shard[self.target_attack_point][self.target_byte]
        metadata = {
            '{}__{}'.format(attack_point, byte): val
            for attack_point in [k for k in shard.keys() if k != 'traces']
            for byte, val in enumerate(shard[attack_point])
        }
        if idx is not None:
            traces = traces[None, idx]
            target = target[None, idx]
            metadata = {key: val[None, idx] for key, val in metadata.items()}
        traces = np.array(traces, dtype=float)
        if traces.shape[1] % self.downsample_ratio > 0:
            padding = -np.inf*np.ones((traces.shape[0], self.downsample_ratio - (traces.shape[1] % self.downsample_ratio), 1))
            traces = np.concatenate([traces, padding], axis=1)
        traces = traces.reshape(-1, traces.shape[1]//self.downsample_ratio, self.downsample_ratio, 1).max(axis=2)
        traces = traces.transpose(0, 2, 1)
        if idx is not None:
            traces = traces.reshape((1, -1))
        return traces, target, metadata
    
    def load_idx(self, idx):
        if self.store_in_memory:
            return super().load_idx(idx)
        else:
            file_idx = np.where(np.logical_and(self.file_lengths[:-1] <= idx, self.file_lengths[1:] > idx))[0][0]
            filename = self.filenames[file_idx]
            shard_idx = idx - self.file_lengths[file_idx]
            traces, target, metadata = self.load_shard(filename, idx=shard_idx)
            if hasattr(self, 'return_metadata') and self.return_metadata:
                return traces, target, metadata
            else:
                return traces, target

AVAILABLE_DATASETS = [GoogleTinyAES]