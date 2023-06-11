import os
import shutil
import wget
import zipfile

# folders in which downloaded resources will be stored
DOWNLOADS = os.path.join('.', 'downloads')
PRETRAINED_MODELS = os.path.join(DOWNLOADS, 'pretrained_models')
DATASETS = os.path.join(DOWNLOADS, 'datasets')
RAW = os.path.join(DOWNLOADS, 'raw')

def verify_folders():
    os.makedirs(DOWNLOADS, exist_ok=True)
    os.makedirs(PRETRAINED_MODELS, exist_ok=True)
    os.makedirs(DATASETS, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

def download(url_list, subdir, force=False):
    if not type(url_list) == list:
        assert type(url_list) == str
        url_list = [url_list]
    for url in url_list:
        filename = os.path.split(url)[-1]
        dest = os.path.join(RAW, subdir, filename)
        if force or not(os.path.exists(dest)):
            if os.path.exists(dest):
                os.remove(dest)
            print('Downloading {} to {} ...'.format(url, dest))
            wget.download(url, dest)
            print()
        else:
            print('Found preexisting download of {} at {} ...'.format(url, dest))

def unzip(subdir):
    base_dir = os.path.join(RAW, subdir)
    for file in os.listdir(base_dir):
        if file.split('.')[-1] == 'zip':
            print('Extracting {} ...'.format(os.path.join(base_dir, file)))
            with zipfile.ZipFile(os.path.join(base_dir, file), 'r') as zip_ref:
                for member in zip_ref.infolist():
                    path = os.path.join(base_dir, member.filename)
                    if not os.path.exists(path):
                        zip_ref.extract(member, base_dir)

def clear_raw(subdir):
    shutil.rmtree(os.path.join(RAW, subdir))
    
def mv(src, dest, src_dir, dest_dir):
    if not os.path.exists(os.path.join(dest_dir, dest)):
        os.rename(os.path.join(src_dir, src), os.path.join(dest_dir, dest))

verify_folders()