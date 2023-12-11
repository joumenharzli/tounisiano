"""
Copyright (c) 2023 Joumen HARZLI
"""

from tounisiano.config import Config
from tounisiano.tasks import GenerateDatasetTask, TrainTask

if __name__ == "__main__":
    config = Config.from_yaml_file("config.yml")
    # task = GenerateDatasetTask(config)
    # task.execute()
    task = TrainTask(config)
    task.execute()
