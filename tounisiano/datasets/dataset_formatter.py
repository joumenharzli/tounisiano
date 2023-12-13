"""
Copyright (c) 2023 Joumen HARZLI
"""

import pandas as pd

from .models import QA, Conversation, Dataset


class DatasetFormatter:
    def __init__(self, system_prompt: str, qa_prompt_format: str) -> None:
        self.system_prompt = system_prompt
        self.qa_prompt_format = qa_prompt_format

    def dataset_to_df(self, dataset: Dataset) -> pd.DataFrame:
        df = pd.DataFrame()
        df["text"] = [self.conversation_to_str(conv) for conv in dataset.conversations]
        df["category"] = dataset.category
        return df

    def conversation_to_str(self, conversation: Conversation) -> str:
        interactions_txt = "".join(
            [self.qa_to_str(interaction) for interaction in conversation.interactions]
        ).rstrip()

        return self.system_prompt + interactions_txt

    def qa_to_str(self, qa: QA) -> str:
        return self.qa_prompt_format.format(user=qa.question, assistant=qa.answer)
