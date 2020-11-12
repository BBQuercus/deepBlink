"""CLI submodule for downloading pre-trained models."""

from urllib import request
import ast
import logging
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

        self.model_url = "https://raw.githubusercontent.com/BBQuercus/deepBlink/master/deepblink/cli/models.txt"

    def __call__(self) -> None:
        """Run check for input image."""
        if self.all:
            self.logger.debug("selected all models for download")
            for name, url in self.models.items():
                self.logger.info(f"Downloading model {name}")
                self.download_model(name, url)

        if self.list:
            print(f"Found the following {len(self.models)} models:")
            for name in self.models.keys():
                print(f"  - {name.capitalize()}")
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
                self.download_model(inp, self.models[inp])
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
        raw_models = request.urlopen(self.model_url).read().decode("utf-8")  # nosec
        models = ast.literal_eval(raw_models)
        self.logger.debug(f"found models {models}")
        return models

    @staticmethod
    def download_model(name: str, url: str):
        """Load a single model."""
        req = requests.get(url, allow_redirects=True)
        open(f"{name}.h5", "wb").write(req.content)
