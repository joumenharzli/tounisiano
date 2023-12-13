"""
Copyright (c) 2023 Joumen HARZLI
"""

import click

from tounisiano.config import Config
from tounisiano.tasks import GenerateDatasetTask, TrainTask


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config-file", default="config.yml", help="Path to the configuration file."
)
def train(config_file):
    config = Config.from_yaml_file(config_file)
    task = TrainTask(config)
    task.execute()


@cli.command()
@click.option(
    "--config-file", default="config.yml", help="Path to the configuration file."
)
def generate_dataset(config_file):
    config = Config.from_yaml_file(config_file)
    task = GenerateDatasetTask(config)
    task.execute()
