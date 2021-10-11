"""CLI submodule for downloading pre-trained models."""

from urllib import request
from urllib.error import URLError
import ast
import logging
import socket
import requests


class HandleDownload:
    """Handle download submodule for CLI.

    Args:
        arg_input: Name of the model.
        arg_list: If the name of all models should be shown.
        arg_all: If all models should be downloaded.
        logger: Verbose logger.
    """

    def __init__(
        self, arg_input: str, arg_list: bool, arg_all: bool, logger: logging.Logger
    ):
        self.input = arg_input
        self.list = arg_list
        self.all = arg_all
        self.logger = logger
        self.logger.info("\U0001F4E5 starting download submodule")

        self.timeout_list = 10
        self.timeout_download = 30
        self.model_url = "https://raw.githubusercontent.com/BBQuercus/deepBlink/master/deepblink/cli/models.txt"

    def __call__(self) -> None:
        """Run check for input image."""
        if self.all:
            self.logger.debug("selected all models for download")
            for name, url in self.models.items():
                self.logger.info(f"Downloading model {name}")
                self.download_model(name, url, self.timeout_download)

        if self.list:
            print(f"Found the following {len(self.models)} models:")
            for name in self.models.keys():
                print(f"  - {name}")
            print(
                (
                    'Download all models using "deepblink download --all" or '
                    'download specific models using "deepblink download --input NAME".'
                )
            )

        if not self.all and self.input is not None:
            inp = self.input.lower()
            if inp in self.models:
                self.logger.info(f"Downloading model {inp}")
                self.download_model(inp, self.models[inp], self.timeout_download)
            else:
                raise ValueError(
                    (
                        f"Model {self.input} does not exist. "
                        'List all available models using "deepblink download --list"'
                    )
                )

    @property
    def models(self) -> dict:
        """Dictionary with all models listed."""
        try:
            req = request.urlopen(self.model_url, timeout=self.timeout_list)  # nosec
        except (URLError, socket.timeout) as e:
            self.logger.debug(f"url timeout occurred: {e}")
            print(
                "Response time was exceeded. "
                "Please wait and try again with a better connection."
            )
        raw_models = req.read().decode("utf-8")
        models = ast.literal_eval(raw_models)
        self.logger.debug(f"found models {models}")
        return models

    @staticmethod
    def download_model(name: str, url: str, timeout: int):
        """Load a single model."""
        try:
            req = requests.get(url, allow_redirects=True, timeout=timeout)
        except requests.exceptions.Timeout:
            print(
                "Response time was exceeded. "
                "Please wait and try again with a better connection."
            )
        open(f"{name}.h5", "wb").write(req.content)
