"""Main / entrypoint function for deepblinks CLI."""

import os

from ._handler import HandleCheck
from ._handler import HandlePredict
from ._handler import HandleTrain
from ._logger import _configure_logger
from ._parser import _parse_args

# Removes tensorflow's information on CPU / GPU availablity.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    """Entrypoint for the CLI."""
    args = _parse_args()
    logger = _configure_logger(args.verbose, args.debug)

    if args.command == "train":
        handler = HandleTrain(arg_config=args.config, arg_gpu=args.gpu, logger=logger)

    if args.command == "check":
        handler = HandleCheck(arg_input=args.INPUT, logger=logger)

    if args.command == "predict":
        handler = HandlePredict(
            arg_model=args.MODEL.name,
            arg_input=args.INPUT,
            arg_output=args.output,
            arg_radius=args.radius,
            arg_type=args.type,
            arg_shape=args.shape,
            logger=logger,
        )

    handler.run()
