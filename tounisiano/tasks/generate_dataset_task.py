"""
Copyright (c) 2023 Joumen HARZLI
"""

from importlib import import_module
from typing import Dict, Type

import pandas as pd

from tounisiano.config import Config
from tounisiano.datasets import BaseDataset, DatasetFormatter


class GenerateDatasetTask:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_formatter = DatasetFormatter(
            system_prompt=config.system_prompt, qa_prompt_format=config.qa_prompt_format
        )

    def execute(self):
        datasets_dfs = []

        for dataset_name in self.config.datasets:
            classes = self.import_and_register_all_classes(dataset_name)
            if classes:
                for name, dataset_class in classes.items():
                    output_path = f"{self.config.datasets_output_dir}{name}.json"
                    dataset_instance = dataset_class()
                    dataset = dataset_instance.generate(output_path)
                    dataset_df = self.dataset_formatter.dataset_to_df(dataset)
                    datasets_dfs.append(dataset_df)

        result_df = pd.concat(datasets_dfs)
        result_df.to_parquet(self.config.merged_dataset_output_path, compression="gzip")

    @staticmethod
    def import_and_register_all_classes(
        module_name: str,
    ) -> Dict[str, Type[BaseDataset]]:
        registered_classes = {}
        try:
            my_module = import_module(module_name)
            for name in dir(my_module):
                obj = getattr(my_module, name)
                if isinstance(obj, type) and issubclass(obj, BaseDataset):
                    registered_classes[obj.__name__] = obj
            return registered_classes
        except ModuleNotFoundError as e:
            print(f"Error: {e}")
            return {}
