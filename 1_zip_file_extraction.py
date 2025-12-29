import zipfile
from tqdm import tqdm
import os

zip_path = "./SDNET2018.zip"
extract_path = "./SDNET2018"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()

    for file in tqdm(file_list, desc="Extracting SDNET2018", unit="file"):
        zip_ref.extract(file, extract_path)

print("✅ Extraction complete")
