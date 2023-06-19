import os

import resources

NAME = 'Zaid'
ASSET_URLS = [
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/ASCAD/N0%3D0/ASCAD_trained_models/ASCAD_desync0',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/ASCAD/N0%3D50/ASCAD_trained_models/ASCAD_desync50',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/ASCAD/N0%3D100/ASCAD_trained_models/ASCAD_desync100',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/DPA-contest%20v4/DPAv4_trained_models/DPA-contest_v4',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/DPA-contest%20v4/DPAv4_dataset.zip',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/AES_HD/AESHD_trained_models/AES_HD',
        r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/AES_RD/AESRD_trained_models/AES_RD'
]
VALID_MODELS = ['AES_HD', 'AES_RD', 'ASCAD_desync0', 'ASCAD_desync100', 'ASCAD_desync50', 'DPA-contest_v4']

def get_dataset_path():
    return os.path.join(resources.DOWNLOADS, NAME, 'DPAv4_dataset')

def get_model_path(model):
    assert model in VALID_MODELS
    return os.path.join(resources.DOWNLOADS, NAME, model)

if not(os.path.exists(get_dataset_path())) or not(all(os.path.exists(get_model_path(model)) for model in VALID_MODELS)):
    resources.download(ASSET_URLS, NAME)
    resources.unzip(NAME)
    resources.move(NAME)
    resources.clear_raw(NAME)