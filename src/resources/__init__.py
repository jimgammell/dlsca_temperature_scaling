import os
import shutil
import requests
from tqdm import tqdm
import zipfile
import numpy as np

# folders in which downloaded resources will be stored
DOWNLOADS = os.path.join('..', 'downloads')
RAW = os.path.join(DOWNLOADS, 'compressed')

def download(url_list, subdir, force=False, chunk_size=2**20):
    if not type(url_list) == list:
        assert type(url_list) == str
        url_list = [url_list]
    os.makedirs(os.path.join(RAW, subdir), exist_ok=True)
    for url in url_list:
        filename = os.path.split(url)[-1]
        dest = os.path.join(RAW, subdir, filename)
        if force or not(os.path.exists(dest)):
            if os.path.exists(dest):
                os.remove(dest)
            print('Downloading {} to {} ...'.format(url, dest))
            response = requests.get(url, stream=True)
            with open(dest, 'wb') as F:
                for data in tqdm(
                    response.iter_content(chunk_size=chunk_size),
                    total=int(np.ceil(int(response.headers['Content-length'])/chunk_size)),
                    unit='MB'
                ):
                    F.write(data)
            print()
        else:
            print('Found preexisting download of {} at {} ...'.format(url, dest))

def unzip(subdir):
    base_dir = os.path.join(RAW, subdir)
    dest_dir = os.path.join(DOWNLOADS, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(base_dir):
        if file.split('.')[-1] == 'zip':
            print('Extracting {} ...'.format(os.path.join(base_dir, file)))
            with zipfile.ZipFile(os.path.join(base_dir, file), 'r') as zip_ref:
                for member in zip_ref.infolist():
                    zip_ref.extract(member, dest_dir)

def clear_raw(subdir):
    shutil.rmtree(os.path.join(RAW, subdir))
    if len(os.listdir(RAW)) == 0:
        shutil.rmtree(RAW)
    
def move(subdir):
    base_dir = os.path.join(RAW, subdir)
    for file in os.listdir(base_dir):
        if not file.split('.')[-1] == 'zip':
            os.makedirs(os.path.join(DOWNLOADS, subdir), exist_ok=True)
            os.rename(os.path.join(base_dir, file), os.path.join(DOWNLOADS, subdir, file))

os.makedirs(DOWNLOADS, exist_ok=True)