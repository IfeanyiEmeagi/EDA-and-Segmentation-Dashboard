"""This module extracts information from your `.env` file so that
"""

import os

# pydantic used for data validation: https://pydantic-docs.helpmanual.io/
from pydantic import BaseSettings


def return_full_path(filename: str = ".filepath"):
    """Uses os to return the correct path of the `.env` file."""
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    full_path = os.path.join(directory_name, filename)
    return full_path


class Settings(BaseSettings):
    """Uses pydantic to define the file path for project."""

    file_path: str

    class Config:
        env_file = return_full_path(".filepath")


# Create instance of `Settings` class that will be imported
settings = Settings()