#!/usr/bin/env python3
import os
import requests
import zipfile
import io
import shutil

# --- Configuration ---
REPO_OWNER = "thephodit"
REPO_NAME = "models_gen_ai"
BRANCH = "main"
FOLDERS_TO_DOWNLOAD = ["mnist_model", "cifar10_model"]

# URL for downloading the repo as a ZIP
ZIP_URL = (
    f"https://github.com/{REPO_OWNER}/{REPO_NAME}/"
    f"archive/refs/heads/{BRANCH}.zip"
)

def download_and_extract_folders(zip_url, folder_names, target_dir="."):
    print("Downloading pretrained models for the notebook to work...")
    print(f"Fetching repository archive from {zip_url}")
    response = requests.get(zip_url)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    root_prefix = f"{REPO_NAME}-{BRANCH}/"
    names_in_zip = zip_file.namelist()

    for folder in folder_names:
        expected_prefix = root_prefix + folder + "/"
        if not any(name.startswith(expected_prefix) for name in names_in_zip):
            print(f"Warning: '{folder}' not found in repository archive.")
            continue

        print(f"Extracting '{folder}'...")
        for member in names_in_zip:
            if member.startswith(expected_prefix):
                rel_path = os.path.relpath(member, root_prefix)
                dest_path = os.path.join(target_dir, rel_path)
                if member.endswith('/'):
                    os.makedirs(dest_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with zip_file.open(member) as src, open(dest_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
        print(f"Done: '{folder}' extracted.")


def is_folder_empty(path):
    return not any(os.scandir(path))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing_folders = []

    for folder in FOLDERS_TO_DOWNLOAD:
        full_path = os.path.join(script_dir, folder)
        if not os.path.isdir(full_path):
            print(f"Folder missing: {full_path}")
            missing_folders.append(folder)
        elif is_folder_empty(full_path):
            print(f"Folder exists but is empty: {full_path}")
            missing_folders.append(folder)
        else:
            print(f"Folder OK: {full_path}")

    if not missing_folders:
        print("All required folders are present and populated.")
        return

    try:
        download_and_extract_folders(ZIP_URL, missing_folders, script_dir)
        print("All missing folders downloaded and extracted.")
    except Exception as e:
        print(f"Error during download/extraction: {e}")


if __name__ == "__main__":
    main()
