import os

import resources

NAME = 'AES-HD'
ASSET_URLS = [
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/labels.csv',
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/traces_1.csv',
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/traces_2.csv',
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/traces_3.csv',
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/traces_4.csv',
    r'https://github.com/AESHD/AES_HD_Dataset/raw/master/traces_5.csv'
]

def get_dataset_path():
    return os.path.join(resources.DOWNLOADS, NAME)

if not os.path.exists(get_dataset_path()):
    resources.download(ASSET_URLS, NAME)
    resources.move(NAME)
    resources.clear_raw(NAME)