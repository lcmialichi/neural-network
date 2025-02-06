import requests
import zipfile
import os
from tqdm import tqdm

url = "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/breast-histopathology-images"
output_path = "./data/breast-histopathology-images.zip"
extract_path = "./data/breast-histopathology-images"

def doing(text: str, end='\r'):
    print(f"\033[1;33m{text}\033[0m", flush=True, end=end)

def done(text: str, end='\n'):
    print(f"\032[1;33m{text}\033[0m", flush=True, end=end)

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, "wb") as file, tqdm(
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

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"\033[1;33mFiles transfered to: {extract_path}\033[0m", flush=True)

def remove_zip(zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("\033[1;33mDone clearing\033[0m", flush=True)

doing('Downloading dataset...')
download_file(url, output_path)
done(f'Downloaded to path: {output_path}')

doing('Extracting dataset...')
extract_zip(output_path, extract_path)
done(f'Extracted to path: {extract_path}')

doing(f'removing {output_path}...')
remove_zip(output_path)
done(f'zip removed!: {extract_path}')

