import os

import resources

NAME = 'ASCADv1-Fixed'
ASSET_URL = r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip'
VALID_ARCH = ['mlp', 'cnn']
VALID_DESYNC = [0, 50, 100]

def get_dataset_path(desync):
    assert desync in VALID_DESYNC
    return os.path.join(
        resources.DOWNLOADS, NAME, 'ASCAD_data', 'ASCAD_databases', 'ASCAD' + ('_desync%d'%(desync) if desync!=0 else '') + '.h5'
    )

def get_model_path(desync, arch):
    assert desync in VALID_DESYNC
    assert arch in VALID_ARCH
    if arch == 'cnn':
        model_name = 'cnn_best_ascad_desync%d_epochs75_classes256_batchsize200.h5'%(desync)
    elif arch == 'mlp':
        model_name = 'mlp_best_ascad_desync%d_node200_layernb6_epochs200_classes256_batchsize100.h5'%(desync)
    return os.path.join(resources.DOWNLOADS, NAME, 'ASCAD_data', 'ASCAD_trained_models', model_name)

if not(all(os.path.exists(get_dataset_path(desync)) for desync in VALID_DESYNC)) or not(all(os.path.exists(get_model_path(desync, arch)) for desync in VALID_DESYNC for arch in VALID_ARCH)):
    resources.download(ASSET_URL, NAME)
    resources.unzip(NAME)
    resources.clear_raw(NAME)