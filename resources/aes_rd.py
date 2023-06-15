import os

import resources

NAME = 'AES-RD'
ASSET_URL = r'https://github.com/ikizhvatov/randomdelays-traces/raw/master/ctraces_fm16x4_2.mat'

def get_dataset_path():
    return os.path.join(resources.DOWNLOADS, NAME)

if not os.path.exists(get_dataset_path()):
    resources.download(ASSET_URL, NAME)
    resources.move(NAME)
    resources.clear_raw(NAME)