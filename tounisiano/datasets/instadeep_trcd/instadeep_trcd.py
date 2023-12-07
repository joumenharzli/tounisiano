"""
Copyright (c) 2023 Joumen HARZLI
"""

import json
from typing import List

from tounisiano.datasets import QA, AbstractDataset, Conversation, Dataset
from tounisiano.utils import utils

DATASETS_URLS = [
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_train.json",
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_valid.json",
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_test.json",
]

CATEGORY = "LAW_AND_GOV"


class InstadeepTRCD(AbstractDataset):
    def do_generate(self) -> Dataset:
        dataset = Dataset(conversations=[], category=CATEGORY)
        for dataset_url in DATASETS_URLS:
            dataset.conversations.extend(
                self.parse_dataset(utils.download_file(dataset_url))
            )
        return dataset

    @staticmethod
    def parse_dataset(file_path) -> List[Conversation]:
        conversations = []

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            for item in data.get("data", []):
                for paragraph in item.get("paragraphs", []):
                    conversation = Conversation(interactions=[])
                    for qa in paragraph.get("qas", []):
                        question_text = (
                            qa.get("question", "").replace("- +", "").strip()
                        )
                        answer_text = qa.get("answers", [{}])[0].get("text", "").strip()

                        conversation.interactions.append(
                            QA(question=question_text, answer=answer_text)
                        )
                    conversations.append(conversation)

        return conversations
