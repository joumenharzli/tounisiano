"""
Copyright (c) 2023 Joumen HARZLI
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict

from tounisiano import utils

from .models import QA, Conversation, Dataset


class BaseDataset(ABC):
    @abstractmethod
    def do_generate(self) -> Dataset:
        pass

    def generate(self, output_path: str) -> Dataset:
        if utils.file_exists(output_path):
            return self._load_dataset_from_json(output_path)

        dataset = self.do_generate()
        self._write_dataset_to_json_file(output_path, asdict(dataset))
        return dataset

    def _load_dataset_from_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        conversations_data = data.get("conversations", [])
        category = data.get("category", "")

        conversations = []
        for conversation_data in conversations_data:
            interactions_data = conversation_data.get("interactions")
            interactions = [
                QA(
                    question=interaction.get("question"),
                    answer=interaction.get("answer"),
                )
                for interaction in interactions_data
            ]
            conversation = Conversation(interactions=interactions)
            conversations.append(conversation)

        return Dataset(conversations=conversations, category=category)

    def _write_dataset_to_json_file(self, output_path, dataset_dict):
        json_data = json.dumps(dataset_dict, indent=2, ensure_ascii=False)
        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_data)
