"""
Copyright (c) 2023 Joumen HARZLI
"""

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class Config:
    datasets_output_dir: str = ""
    datasets: List[str] = field(default_factory=list)
    system_prompt: str = ""
    qa_prompt_format: str = ""

    @classmethod
    def from_yaml_file(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return cls(**data)
