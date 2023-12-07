"""
Copyright (c) 2023 Joumen HARZLI
"""

import os
import tempfile

import requests


def file_exists(file_path):
    return os.path.exists(file_path)


def download_file(url):
    destination = tempfile.NamedTemporaryFile(delete=False).name
    response = requests.get(url, timeout=60)
    with open(destination, "wb") as file:
        file.write(response.content)
    return destination
