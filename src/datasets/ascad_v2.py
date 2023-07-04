import os
import h5py import numpy as np

import resources
from datasets.common import DatasetBase

class ASCADv2(DatasetBase):
    def __init__(self, train=True, resource_path=None, **kwargs):
        if resource_path is None:
            from resources import ascadv2
            resource_path = ascadv2.get_dataset_path()
        