import os
import h5py
import numpy as np
from tqdm import tqdm
import pickle

import resources

NAME = 'Google-TinyAES'
ASSET_URL = r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'

def get_dataset_path(train=True):
    return os.path.join(
        resources.DOWNLOADS, NAME, 'datasets', 'tinyaes', 'train' if train else 'test'
    )

# Convert the dataset to hdf5 format, which is about 500x faster on my system than loading a different
#   npz file for every trace that is trained on.
def decompress_dataset():
    print('Decompressing files in {} ...'.format(os.path.join(resources.DOWNLOADS, NAME)))
    for train in [True, False]:
        path = get_dataset_path(train)
        with h5py.File(os.path.join(path, 'data.hdf5'), 'w') as F:
            F.create_dataset('traces', (65536, 1, 80000), dtype=np.float16)
            files = [f for f in os.listdir(path) if f.split('.')[-1] == 'npz']
            for f_idx, filename in tqdm(enumerate(files), total=len(files)):
                shard = np.load(os.path.join(path, filename))
                traces = np.array(shard['traces'], dtype=np.float16).transpose((0, 2, 1))
                F['traces'][f_idx*256 : (f_idx+1)*256, :, :] = traces
                metadata = {
                    '{}__{}'.format(attack_point, byte): np.array(val, dtype=np.uint8)
                    for attack_point in shard.keys() if attack_point != 'traces'
                    for byte, val in enumerate(shard[attack_point])
                }
                for key, val in metadata.items():
                    if not key in F.keys():
                        F.create_dataset(key, (65536,), dtype=np.uint8)
                    F[key][f_idx*256 : (f_idx+1)*256] = val
                os.remove(os.path.join(path, filename))

if not all(os.path.exists(os.path.join(get_dataset_path(train), 'data.hdf5')) for train in [True, False]):
    resources.download(ASSET_URL, NAME)
    resources.unzip(NAME)
    resources.clear_raw(NAME)
    decompress_dataset()