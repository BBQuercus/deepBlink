"""CLI submodule for launching the napari prediction viewer."""

import logging


class HandleView:
    """Handle view submodule for CLI.

    Args:
        arg_input: Optional path to image or folder with images.
        arg_model: Optional path to a deepBlink .h5 model.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_input: str, arg_model: str, logger: logging.Logger):
        self.arg_input = arg_input
        self.arg_model = arg_model
        self.logger = logger
        self.logger.info("\U0001F50D starting view submodule")

    def __call__(self):
        """Launch the napari prediction viewer."""
        try:
            from ..napari_plugin.viewer import launch_viewer
        except ImportError as exc:
            self.logger.error(
                "napari is required for the viewer plugin. "
                'Install with: pip install "deepblink[napari]"'
            )
            raise SystemExit(1) from exc

        self.logger.info("launching napari with deepBlink viewer")
        launch_viewer(path=self.arg_input, model=self.arg_model)
