import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())
import h5py
import numpy as np

from datasets.common import DatasetBase

class ASCADv2(DatasetBase):
    def __init__(
        self,
        train=True,
        resource_path=None,
        **kwargs
    ):
        if resource_path is None:
            from resources import ascadv2
            resource_path = ascadv2.get_dataset_path()
        self.resource_path = resource_path
        
        self.name = 'ASCADv2'
        self.data_mean = None
        self.data_stdev = None
        self.train = train
        
        #super().__init__(train=train, **kwargs)
        
    def load_idx(self, idx):
        database_file = h5py.File(self.resource_path, 'r')
        if self.train:
            group = database_file['Attack_traces']
        else:
            group = database_file['Profiling_traces']
        data = group['traces'][idx, :]
        data = np.expand_dims(data, axis=0)
        target = group['labels'][idx]
        alpha_mask = target['alpha_mask']
        beta_mask = target['beta_mask']
        sbox_masked = target['sbox_masked']
        sbox_masked_with_perm = target['sbox_masked_with_perm']
        perm_index = target['perm_index']
        metadata = group['metadata'][idx]
        metadata_rv = {
            'masks': metadata['masks'],
            'plaintext': metadata['plaintext'],
            'key': metadata['key']
        }
            

if __name__ == '__main__':
    dataset = ASCADv2()
    dataset.load_idx(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    