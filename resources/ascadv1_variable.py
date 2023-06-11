import os

import resources

NAME = 'ASCADv1-Variable'
ASSET_URLS = [
    r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5',
    r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-084119/ascad-variable-desync50.h5',
    r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-084306/ascad-variable-desync100.h5',
    r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190801-132322/cnn2-ascad-desync0.h5',
    r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190801-132406/cnn2-ascad-desync50.h5'
]
DATASET_DIR = os.path.join(resources.DATASETS, NAME)
MODELS_DIR = os.path.join(resources.PRETRAINED_MODELS, NAME)
VALID_DESYNC = [0, 50]

def install_assets():
    src_dir = os.path.join(resources.RAW, NAME)
    resources.mv('ascad-variable.h5', 'dataset_desync0.h5', src_dir, DATASET_DIR)
    resources.mv('ascad-variable-desync50.h5', 'dataset_desync50.h5', src_dir, DATASET_DIR)
    resources.mv('ascad-variable-desync100.h5', 'dataset_desync100.h5', src_dir, DATASET_DIR)
    resources.mv('cnn2-ascad-desync0.h5', 'cnn_desync0.h5', src_dir, MODELS_DIR)
    resources.mv('cnn2-ascad-desync50.h5', 'cnn_desync50.h5', src_dir, MODELS_DIR)

def get_dataset_path(desync):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    return os.path.join(DATASET_DIR, 'dataset_desync%d.h5'%(desync))

def get_model_path(desync):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    return os.path.join(MODELS_DIR, 'cnn_desync%d.h5'%(desync))

if not(all(os.path.exists(get_dataset_path(desync)) for desync in VALID_DESYNC)) or not(all(os.path.exists(get_model_path(desync)) for desync in VALID_DESYNC)):
    resources.download(ASSET_URLS, NAME)
    install_assets()
    resources.clear_raw(NAME)