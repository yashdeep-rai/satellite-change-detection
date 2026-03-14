import os
import gdown
import zipfile
import glob

FOLDER_URL = "https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim"
DATASET_DIR = "dataset"

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_dataset():

    if os.path.exists(DATASET_DIR):
        print("Dataset already prepared.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("Downloading LEVIR-CD dataset folder...")

    gdown.download_folder(
        FOLDER_URL,
        output="dataset_raw",
        quiet=False,
        use_cookies=False
    )

    print("Extracting dataset...")

    zip_files = glob.glob("dataset_raw/*.zip")

    for z in zip_files:
        print(f"Extracting {z}")
        extract_zip(z, DATASET_DIR)

    print("Cleaning temporary files...")

    for z in zip_files:
        os.remove(z)

    os.rmdir("dataset_raw")

    print("Dataset ready!")

if __name__ == "__main__":
    download_dataset()
