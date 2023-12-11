"""
Copyright (c) 2023 Joumen HARZLI
"""

import os
import tempfile

import requests


def create_dir(output_directory: str):
    os.makedirs(output_directory, exist_ok=True)


def file_exists(file_path):
    return os.path.exists(file_path)


def download_file(url):
    destination = tempfile.NamedTemporaryFile(delete=False).name
    response = requests.get(url, timeout=60)
    with open(destination, "wb") as file:
        file.write(response.content)
    return destination
