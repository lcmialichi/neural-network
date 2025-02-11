import requests
import zipfile
import os
from tqdm import tqdm

url = "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/breast-histopathology-images"
path = os.path.dirname(os.path.abspath(__file__)) + '/data/breast-histopathology-images'
zip = f"{path}.zip"

def doing(text: str, end='\r'):
    print(f"\033[1;33m{text}\033[0m", flush=True, end=end)

def done(text: str, end='\n'):
    print(f"\033[1;32m{text}\033[0m", flush=True, end=end)

def download_file(url, zip):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        ncols=100,
        leave=False
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)

def extract_zip(zip_path, path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)

def remove_zip(zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)

doing('Downloading dataset...')
download_file(url, zip)
done(f'Downloaded to path: {zip}')

doing('Extracting dataset...')
extract_zip(zip, path)
done(f'Extracted to path: {path}')

doing(f'removing {zip}...')
remove_zip(zip)
done(f'zip removed!: {path}')

