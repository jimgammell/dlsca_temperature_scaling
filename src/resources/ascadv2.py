import os

import resources

NAME = 'ASCADv2'
ASSET_URL = r'https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5'

def get_dataset_path():
    return os.path.join(
        resources.DOWNLOADS, NAME, 'ascadv2-extracted.h5'
    )

if not os.path.exists(get_dataset_path()):
    resources.download(ASSET_URL, NAME)
    resources.move(NAME)
    resources.clear_raw(NAME)