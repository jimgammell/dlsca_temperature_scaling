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
VALID_DESYNC = [0, 50]

def get_dataset_path(desync):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    return os.path.join(resources.DOWNLOADS, NAME, 'ascad-variable' + ('-desync%d'%(desync) if desync!=0 else '') + '.h5')

def get_model_path(desync):
    if not desync in VALID_DESYNC:
        raise ValueError('Invalid desync: {}'.format(desync))
    return os.path.join(resources.DOWNLOADS, NAME, 'cnn2-ascad-desync%d.h5'%(desync))

if not(all(os.path.exists(get_dataset_path(desync)) for desync in VALID_DESYNC)) or not(all(os.path.exists(get_model_path(desync)) for desync in VALID_DESYNC)):
    resources.download(ASSET_URLS, NAME)
    resources.move(NAME)
    resources.clear_raw(NAME)