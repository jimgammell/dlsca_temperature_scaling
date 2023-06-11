import os

import resources

NAME = 'ASCADv1-Fixed'
ASSET_URL = r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip'
DATASET_DIR = os.path.join(resources.DATASETS, NAME)
MODELS_DIR = os.path.join(resources.PRETRAINED_MODELS, NAME)
VALID_ARCH = ['mlp', 'cnn']
VALID_DESYNC = [0, 50, 100]
    
def install_assets():
    dataset_src_dir = os.path.join(resources.RAW, NAME, 'ASCAD_data', 'ASCAD_databases')
    resources.mv('ASCAD.h5', 'dataset_desync0.h5', dataset_src_dir, DATASET_DIR)
    resources.mv('ASCAD_desync50.h5', 'dataset_desync50.h5', dataset_src_dir, DATASET_DIR)
    resources.mv('ASCAD_desync100.h5', 'dataset_desync100.h5', dataset_src_dir, DATASET_DIR)
    
    model_src_dir = os.path.join(resources.RAW, NAME, 'ASCAD_data', 'ASCAD_trained_models')
    resources.mv('cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5', 'cnn_desync0.h5', model_src_dir, MODELS_DIR)
    resources.mv('cnn_best_ascad_desync50_epochs75_classes256_batchsize200.h5', 'cnn_desync50.h5', model_src_dir, MODELS_DIR)
    resources.mv('cnn_best_ascad_desync100_epochs75_classes256_batchsize200.h5', 'cnn_desync100.h5', model_src_dir, MODELS_DIR)
    resources.mv('mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5', 'mlp_desync0.h5', model_src_dir, MODELS_DIR)
    resources.mv('mlp_best_ascad_desync50_node200_layernb6_epochs200_classes256_batchsize100.h5', 'mlp_desync50.h5', model_src_dir, MODELS_DIR)
    resources.mv('mlp_best_ascad_desync50_node200_layernb6_epochs200_classes256_batchsize100.h5', 'mlp_desync100.h5', model_src_dir, MODELS_DIR)

def get_dataset_path(desync):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    return os.path.join(DATASET_DIR, 'dataset_desync%d.h5'%(desync))

def get_model_path(desync, arch):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    if not arch in VALID_ARCH:
        raise ValueError('Invalid architecture: {}'.format(arch))
    return os.path.join(MODELS_DIR, '%s_desync%d.h5'%(arch, desync))

if not(all(os.path.exists(get_dataset_path(desync)) for desync in VALID_DESYNC)) or not(all(os.path.exists(get_model_path(desync, arch)) for desync in VALID_DESYNC for arch in VALID_ARCH)):
    resources.download(ASSET_URL, NAME)
    resources.unzip(NAME)
    install_assets()
    resources.clear_raw(NAME)