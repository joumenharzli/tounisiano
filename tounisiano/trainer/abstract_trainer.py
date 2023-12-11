"""
Copyright (c) 2023 Joumen HARZLI
"""

from abc import ABC, abstractmethod

from tounisiano.trainer.models import TrainingParameters


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, merged_dataset_output_path: str, params: TrainingParameters):
        pass
