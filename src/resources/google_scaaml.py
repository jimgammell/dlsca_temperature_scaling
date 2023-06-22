import os

import resources

NAME = 'Google-TinyAES'
ASSET_URL = r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'

def get_dataset_path(train=True):
    return os.path.join(
        resources.DOWNLOADS, NAME, 'datasets', 'tinyaes', 'train' if train else 'test'
    )

if not all(os.path.exists(get_dataset_path(train)) for train in [True, False]):
    resources.download(ASSET_URL, NAME)
    resources.unzip(NAME)
    resources.clear_raw(NAME)