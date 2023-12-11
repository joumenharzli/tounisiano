"""
Copyright (c) 2023 Joumen HARZLI
"""

from dataclasses import dataclass
from typing import List

from dataclasses_json import Undefined, dataclass_json


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class LoRAFineTuningParameters:
    rate: int
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class TrainingParameters:
    base_model: str
    outputs_path: str
    eos_token: str
    new_tokens: List[str]
    max_seq_length: int
    lora: LoRAFineTuningParameters
    epochs: int
    batch_size: int
    learning_rate: float
