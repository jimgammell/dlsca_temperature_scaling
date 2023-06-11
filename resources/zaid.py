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
DATASET_DIR = os.path.join(resources.DATASETS, NAME+'_DPAv4-subset')
MODELS_DIR = os.path.join(resources.PRETRAINED_MODELS, NAME)
VALID_DATASETS = ['ASCADv1-Fixed', 'DPAv4', 'AES-HD', 'AES-RD']
VALID_DESYNC = [0, 50, 100]

def install_assets():
    dpa_src_dir = os.path.join(resources.RAW, NAME, 'DPAv4_dataset')
    resources.mv('attack_labels_dpav4.npy', 'attack_labels.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('attack_offset_dpav4.npy', 'attack_offset.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('attack_plaintext_dpav4.npy', 'attack_plaintext.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('attack_traces_dpav4.npy', 'attack_traces.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('key.npy', 'key.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('mask.npy', 'mask.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('profiling_labels_dpav4.npy', 'profiling_labels.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('profiling_plaintext_dpav4.npy', 'profiling_plaintext.npy', dpa_src_dir, DATASET_DIR)
    resources.mv('profiling_traces_dpav4.npy', 'profiling_traces.npy', dpa_src_dir, DATASET_DIR)
    
    model_src_dir = os.path.join(resources.RAW, NAME)
    resources.mv('AES_HD', 'AES-HD', model_src_dir, MODELS_DIR)
    resources.mv('AES_RD', 'AES-RD', model_src_dir, MODELS_DIR)
    resources.mv('ASCAD_desync0', 'ASCADv1-Fixed-Desync0', model_src_dir, MODELS_DIR)
    resources.mv('ASCAD_desync50', 'ASCADv1-Fixed-Desync50', model_src_dir, MODELS_DIR)
    resources.mv('ASCAD_desync100', 'ASCADv1-Fixed-Desync100', model_src_dir, MODELS_DIR)
    resources.mv('DPA-contest_v4', 'DPAv4', model_src_dir, MODELS_DIR)

def get_dataset_path():
    return os.path.join(DATASET_DIR)

def get_model_path(dataset, desync=None):
    if not dataset in VALID_DATASETS:
        raise ValueError('Invalid dataset: {}'.format(dataset))
    if (desync is not None) and not(desync in VALID_DESYNC):
        raise ValueError('Invalid desync: {}'.format(desync))
    if desync is None:
        return os.path.join(MODELS_DIR, dataset)
    else:
        return os.path.join(MODELS_DIR, '%s_Desync%d'%(dataset, desync))

if not(os.path.exists(get_dataset_path())) or not(all(os.path.exists(get_model_path(model)) for model in VALID_DATASETS if model is not 'ASCADv1-Fixed')) or not(all(os.path.exists(get_model_path('ASCADv1-Fixed', desync)) for desync in VALID_DESYNC)):
    resources.download(ASSET_URLS, NAME)
    resources.unzip(NAME)
    install_assets()
    resources.clear_raw(NAME)