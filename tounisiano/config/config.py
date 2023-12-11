"""
Copyright (c) 2023 Joumen HARZLI
"""

from dataclasses import dataclass, field
from typing import List

import yaml
from dataclasses_json import Undefined, dataclass_json

from tounisiano.trainer.models import TrainingParameters


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Config:
    datasets_output_dir: str = ""
    merged_dataset_output_path: str = ""
    datasets: List[str] = field(default_factory=list)
    system_prompt: str = ""
    qa_prompt_format: str = ""
    training: TrainingParameters = None

    @classmethod
    def from_yaml_file(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
            return Config.from_dict(yaml_data)
