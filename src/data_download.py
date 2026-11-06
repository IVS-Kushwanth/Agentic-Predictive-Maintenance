"""Data Acquisition Module for Agentic Predictive Maintenance Framework.

Downloads and extracts:
1. NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
2. UCI AI4I 2020 Predictive Maintenance Dataset
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with progress reporting and user-agent header."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Destination: {dest_path}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", -1))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)", end="")
            else:
                print(f"\rDownloaded: {downloaded / (1024*1024):.2f} MB", end="")
    print("\nDownload complete.")
    return dest_path


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract all files from a ZIP archive."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def download_nasa_cmapss(config: dict, force: bool = False) -> None:
    """Download and unpack the NASA C-MAPSS dataset."""
    raw_dir = Path(config["paths"]["raw_nasa_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        f"{split}_{subset}.txt"
        for subset in config["nasa"]["subsets"]
        for split in ["train", "test", "RUL"]
    ]

    all_exist = all((raw_dir / f).exists() for f in expected_files)
    if all_exist and not force:
        print("All NASA C-MAPSS raw files are already present. Skipping download.")
        return

    url = config["downloads"]["nasa_url"]
    temp_zip = raw_dir / "cmapss_download.zip"
    download_file(url, temp_zip)

    # Extract to a temp directory to inspect structure
    temp_extract = raw_dir / "_temp_extract"
    extract_zip(temp_zip, temp_extract)

    # Search for files recursively and place in raw_dir
    found_files = list(temp_extract.rglob("*.txt"))
    # Also check if there is an inner zip file (often C-MAPSS has CMAPSSData.zip inside)
    inner_zips = list(temp_extract.rglob("*.zip"))
    for iz in inner_zips:
        print(f"Found nested archive {iz.name}, extracting...")
        with zipfile.ZipFile(iz, "r") as inner_ref:
            inner_ref.extractall(temp_extract)

    # Search again for text files
    found_files = list(temp_extract.rglob("*.txt"))
    for file_path in found_files:
        dest_file = raw_dir / file_path.name
        shutil.copy2(file_path, dest_file)
        print(f"Copied: {dest_file.name} -> {dest_file}")

    # Clean up temporary zip and temp extraction folder
    if temp_zip.exists():
        temp_zip.unlink()
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    print("NASA C-MAPSS dataset ready.")


def download_ai4i(config: dict, force: bool = False) -> None:
    """Download and unpack the UCI AI4I 2020 dataset."""
    raw_dir = Path(config["paths"]["raw_ai4i_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected_file = raw_dir / config["ai4i"]["raw_file"]
    if expected_file.exists() and not force:
        print(f"AI4I dataset file {expected_file.name} already exists. Skipping download.")
        return

    url = config["downloads"]["ai4i_url"]
    temp_zip = raw_dir / "ai4i_download.zip"
    try:
        download_file(url, temp_zip)
        temp_extract = raw_dir / "_temp_extract"
        extract_zip(temp_zip, temp_extract)

        # Look for ai4i2020.csv
        csv_candidates = list(temp_extract.rglob("*.csv"))
        if csv_candidates:
            shutil.copy2(csv_candidates[0], expected_file)
            print(f"Copied {csv_candidates[0].name} to {expected_file}")
        else:
            print("Warning: No CSV file found in archive. Checking archive contents.")

        # Clean up
        if temp_zip.exists():
            temp_zip.unlink()
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

    except Exception as e:
        print(f"Error downloading from primary UCI URL: {e}")
        # Secondary fallback if UCI zip structure changed
        alt_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        print(f"Attempting fallback direct CSV download from: {alt_url}")
        download_file(alt_url, expected_file)

    print("UCI AI4I 2020 dataset ready.")


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    config = load_config()

    print("=" * 60)
    print("STEP 1: NASA C-MAPSS DATASET ACQUISITION")
    print("=" * 60)
    download_nasa_cmapss(config)

    print("\n" + "=" * 60)
    print("STEP 2: UCI AI4I 2020 DATASET ACQUISITION")
    print("=" * 60)
    download_ai4i(config)

    print("\nDataset acquisition complete.")


if __name__ == "__main__":
    main()
