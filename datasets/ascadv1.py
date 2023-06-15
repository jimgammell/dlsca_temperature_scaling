import os
import h5py
import numpy as np

import resources
import datasets

class ASCADv1(datasets.DatasetBase):
    def __init__(self, desync=0, variable=False, train=True, resource_path=None, **kwargs):
        if resource_path is None:
            if variable:
                from resources import ascadv1_variable
                assert desync in ascadv1_variable.VALID_DESYNC
                resource_path = ascadv1_variable.get_dataset_path(desync)
            else:
                from resources import ascadv1_fixed
                assert desync in ascadv1_fixed.VALID_DESYNC
                resource_path = ascadv1_fixed.get_dataset_path(desync)
        database_file = h5py.File(resource_path, 'r')
        dbidx_prefix = 'Profiling_traces' if train else 'Attack_traces'
        self.data = np.array(database_file[dbidx_prefix]['traces'], dtype=float)
        self.data = np.expand_dims(self.data, axis=1)
        self.targets = np.array(database_file[dbidx_prefix]['labels'], dtype=np.uint8)
        self.metadata = {
            'plaintext': np.array(database_file[dbidx_prefix]['metadata']['plaintext'], dtype=np.uint8),
            'key': np.array(database_file[dbidx_prefix]['metadata']['key'], dtype=np.uint8),
            'masks': np.array(database_file[dbidx_prefix]['metadata']['masks'], dtype=np.uint8)
        }
        self.name = 'ASCADv1%s_Desync%d'%('Variable' if variable else 'Fixed', desync)
        
        super().__init__(train=train, **kwargs)