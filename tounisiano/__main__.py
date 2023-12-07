"""
Copyright (c) 2023 Joumen HARZLI
"""

from tounisiano.config.config import Config
from tounisiano.tasks.generate_dataset import GenerateDatasetTask

if __name__ == "__main__":
    config = Config.from_yaml_file("config.yml")
    dataset_output_file = f"{config.datasets_output_dir}dataset.parquet.gzip"
    task = GenerateDatasetTask(config)
    task.execute(dataset_output_file)
