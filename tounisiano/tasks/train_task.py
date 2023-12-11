"""
Copyright (c) 2023 Joumen HARZLI
"""


from tounisiano.config import Config
from tounisiano.trainer import QLoRAFineTuning


class TrainTask:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.trainer = QLoRAFineTuning()

    def execute(self):
        self.trainer.train(self.config.merged_dataset_output_path, self.config.training)
