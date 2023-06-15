import os
import numpy as np

import resources
import datasets

class AES_HD(datasets.DatasetBase):
    def __init__(self, resource_path=None, **kwargs):
        if resource_path is None:
            from resources import aes_hd
            resource_path = aes_hd.get_dataset_path()
        self.data = np.concatenate([
            np.loadtxt(os.path.join(resource_path, 'traces_%d.csv'%(idx))) for idx in range(1, 6)
        ])
        self.targets = np.loadtxt(os.path.join(resource_path, 'labels.csv'))
        self.name = 'AES-HD'
        
        super().__init__(**kwargs)