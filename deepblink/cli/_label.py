"""CLI submodule for launching the napari labeling plugin."""

import logging


class HandleLabel:
    """Handle labeling submodule for CLI.

    Args:
        arg_input: Optional path to folder with images.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_input: str, logger: logging.Logger):
        self.arg_input = arg_input
        self.logger = logger
        self.logger.info("\U0001F3F7 starting labeling submodule")

    def __call__(self):
        """Launch the napari labeling widget."""
        try:
            from ..napari_plugin.widget import launch
        except ImportError as exc:
            self.logger.error(
                "napari is required for the labeling plugin. "
                'Install with: pip install "deepblink[napari]"'
            )
            raise SystemExit(1) from exc

        self.logger.info("launching napari with deepBlink labeler")
        launch(path=self.arg_input)
