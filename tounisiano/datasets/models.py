"""
Copyright (c) 2023 Joumen HARZLI
"""

from dataclasses import dataclass
from typing import List


@dataclass
class QA:
    question: str
    answer: str


@dataclass
class Conversation:
    interactions: List[QA]


@dataclass
class Dataset:
    conversations: List[Conversation]
    category: str
