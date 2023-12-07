"""
Copyright (c) 2023 Joumen HARZLI
"""

import requests
import tempfile
import os


def file_exists(file_path):
    return os.path.exists(file_path)


def download_file(url):
    destination = tempfile.NamedTemporaryFile(delete=False).name
    response = requests.get(url)
    with open(destination, "wb") as file:
        file.write(response.content)
    return destination
