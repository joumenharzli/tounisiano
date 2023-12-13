"""
Copyright (c) 2023 Joumen HARZLI
"""

from abc import ABC, abstractmethod

from .models import TrainingParameters


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, merged_dataset_output_path: str, params: TrainingParameters):
        pass
