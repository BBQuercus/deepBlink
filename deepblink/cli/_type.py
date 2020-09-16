"""Custom argparse types / classes."""

import argparse
import os
import re


class FileFolderType:
    """Custom type supporting folders or files."""

    def __init__(self):
        pass

    def __call__(self, value):  # noqa: D102
        """Python type internal function called by argparse to check input."""
        if not any((os.path.isdir(value), os.path.isfile(value))):
            raise argparse.ArgumentTypeError(
                f"Input value must be file or folder. '{value}' is not."
            )
        return value


class FolderType:
    """Custom type supporting folders."""

    def __init__(self):
        pass

    def __call__(self, value):  # noqa: D102
        """Python type internal function called by argparse to check input."""
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError(
                f"Input value must be folder and must exist. '{value}' is not."
            )
        return value


class ShapeType:
    """Custom type for image shapes."""

    def __init__(self):
        self.remove_characters = ["(", ")", " "]
        self.allowed_characters = "xy3ctz"
        self.required_characters = ["x", "y"]

    def __call__(self, value):  # noqa: D102
        """Python type internal function called by argparse to check input."""
        if not isinstance(value, str):
            raise ValueError(f"Input value must be a string. '{value}' is not.")

        raw_value = value
        for c in self.remove_characters:
            value = value.replace(c, "")
        if not bool(re.match(f"^[{self.allowed_characters},]+$", value)):
            raise ValueError(
                f"Input must only contain values '{self.allowed_characters},'. '{raw_value}' does not."
            )
        if not bool(
            re.match(
                f"^([{self.allowed_characters}],)+[{self.allowed_characters}]$", value
            )
        ):
            raise ValueError(
                f"Input must have format '(?,?,?,?)'. '{raw_value}' does not."
            )
        if not all([c in value for c in self.required_characters]):
            raise ValueError(
                f"Input must contain {self.required_characters}. '{raw_value}' does not."
            )
        return raw_value
