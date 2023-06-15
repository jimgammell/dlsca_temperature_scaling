import os
import numpy as np
from scipy.io import loadmat

import resources
import datasets

class AES_RD(datasets.DatasetBase):
    def __init__(self, resource_path=None, **kwargs):
        if resource_path is None:
            from resources import aes_rd
            resource_path = aes_rd.get_dataset_path()
        data_file = loadmat(os.path.join(resource_path, 'ctraces_fm16x4_2.mat'))
        self.data = np.array(data_file['CompressedTraces'], dtype=float)
        self.metadata = {
            'plaintext': np.array(data_file['plaintext'], dtype=np.uint8)
        }
        key = np.array(
            [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8
        )
        self.metadata['key'] = np.concatenate([key for _ in range(len(self.data))])
        