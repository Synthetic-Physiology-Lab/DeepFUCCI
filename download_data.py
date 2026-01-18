#!/usr/bin/env python3
"""
Download DeepFUCCI training data and pretrained models from Zenodo.

Usage:
    python download_data.py                    # Download all (data + models)
    python download_data.py --data-only        # Download training data only
    python download_data.py --models-only      # Download pretrained models only
    python download_data.py --list             # List available files without downloading

Requirements:
    pip install requests tqdm
"""

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# TODO: Update these Zenodo record IDs when records are published
ZENODO_RECORDS = {
    "models": {
        "record_id": "16574478",  # TODO: Verify this is the correct published record ID
        "description": "Pretrained StarDist, InstanSeg, and Cellpose-SAM models",
        "extract_to": Path.home() / "models",
        # For draft/preview records, add access token (remove for published records)
        # "token": "your_access_token_here",
    },
    # TODO: Uncomment and configure when training data is uploaded to Zenodo
    # "training_data": {
    #     "record_id": "XXXXXXX",
    #     "description": "FUCCI training dataset (images, masks, classes)",
    #     "extract_to": Path("training_data"),
    # },
}

ZENODO_API_BASE = "https://zenodo.org/api/records"


def get_record_metadata(record_id: str, token: str = None) -> dict:
    """Fetch record metadata from Zenodo API."""
    url = f"{ZENODO_API_BASE}/{record_id}"
    params = {}
    if token:
        params["access_token"] = token

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def download_file(url: str, dest_path: Path, token: str = None, expected_size: int = None, expected_checksum: str = None) -> None:
    """Download a file with progress bar and optional verification."""
    params = {}
    if token:
        params["access_token"] = token

    # Stream download with progress bar
    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", expected_size or 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Verify checksum if provided
    if expected_checksum:
        actual_checksum = compute_md5(dest_path)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Checksum mismatch for {dest_path.name}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        print(f"  Checksum verified: {dest_path.name}")


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract zip archive."""
    print(f"Extracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_to)

    print(f"  Extracted to {extract_to}")


def list_files(record_id: str, token: str = None) -> list:
    """List files in a Zenodo record."""
    metadata = get_record_metadata(record_id, token)
    files = metadata.get("files", [])
    return files


def download_record(record_config: dict, download_dir: Path, extract: bool = True) -> None:
    """Download all files from a Zenodo record."""
    record_id = record_config["record_id"]
    token = record_config.get("token")
    extract_to = record_config.get("extract_to")
    description = record_config.get("description", "")

    print(f"\nDownloading: {description}")
    print(f"  Record ID: {record_id}")

    metadata = get_record_metadata(record_id, token)
    files = metadata.get("files", [])

    if not files:
        print("  No files found in record")
        return

    for file_info in files:
        filename = file_info["key"]
        file_url = file_info["links"]["self"]
        file_size = file_info.get("size")
        file_checksum = file_info.get("checksum", "").replace("md5:", "")

        dest_path = download_dir / filename

        # Skip if already downloaded and checksum matches
        if dest_path.exists() and file_checksum:
            if compute_md5(dest_path) == file_checksum:
                print(f"  Skipping {filename} (already downloaded)")
                continue

        print(f"  Downloading {filename}...")
        download_file(
            file_url,
            dest_path,
            token=token,
            expected_size=file_size,
            expected_checksum=file_checksum if file_checksum else None,
        )

        # Extract if it's a zip file and extraction is requested
        if extract and filename.endswith(".zip") and extract_to:
            extract_archive(dest_path, extract_to)


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepFUCCI data and models from Zenodo"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Download training data only",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Download pretrained models only",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files without downloading",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("downloads"),
        help="Directory to store downloaded files (default: downloads/)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract zip files after download",
    )
    args = parser.parse_args()

    # Determine which records to process
    if args.data_only:
        records_to_process = {k: v for k, v in ZENODO_RECORDS.items() if "data" in k}
    elif args.models_only:
        records_to_process = {k: v for k, v in ZENODO_RECORDS.items() if "model" in k}
    else:
        records_to_process = ZENODO_RECORDS

    if not records_to_process:
        print("No records configured for the selected option.")
        return

    # List mode
    if args.list:
        for name, config in records_to_process.items():
            print(f"\n{name}: {config.get('description', '')}")
            print(f"  Record ID: {config['record_id']}")
            try:
                files = list_files(config["record_id"], config.get("token"))
                for f in files:
                    size_mb = f.get("size", 0) / (1024 * 1024)
                    print(f"    - {f['key']} ({size_mb:.1f} MB)")
            except requests.exceptions.HTTPError as e:
                print(f"    Error fetching record: {e}")
        return

    # Download mode
    args.download_dir.mkdir(parents=True, exist_ok=True)

    for name, config in records_to_process.items():
        try:
            download_record(
                config,
                args.download_dir,
                extract=not args.no_extract,
            )
        except requests.exceptions.HTTPError as e:
            print(f"Error downloading {name}: {e}")
            print("  The record may not be published yet or requires an access token.")

    print("\nDownload complete!")
    print(f"Files saved to: {args.download_dir.absolute()}")


if __name__ == "__main__":
    main()
