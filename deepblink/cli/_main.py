"""Main / entrypoint function for deepblinks CLI."""

import argparse
import os

from ._check import HandleCheck
from ._check import _parse_args_check
from ._config import HandleConfig
from ._config import _parse_args_config
from ._create import HandleCreate
from ._create import _parse_args_create
from ._logger import _configure_logger
from ._parseutil import CustomFormatter
from ._parseutil import _add_utils
from ._predict import HandlePredict
from ._predict import _parse_args_predict
from ._train import HandleTrain
from ._train import _parse_args_train

# Removes tensorflow's information on CPU / GPU availablity.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def arg_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description="deepBlink's CLI \U0001F469\U0000200D\U0001F4BB for all of your blobbing needs.",
        epilog=(
            'See "deepblink <command> --help" to read about the specific subcommand.\n'
            'See the online wiki at "https://github.com/BBQuercus/deepBlink/wiki" for an overview.\n'
            "We hope you enjoy using deepBlink \U0001F603"
        ),
        formatter_class=CustomFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(
        title="Commands for various situations", dest="command",
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    _parse_args_check(subparsers, parent_parser)
    _parse_args_config(subparsers, parent_parser)
    _parse_args_create(subparsers, parent_parser)
    _parse_args_predict(subparsers, parent_parser)
    _parse_args_train(subparsers, parent_parser)
    _add_utils(parser)

    return parser


def main():
    """Entrypoint for the CLI."""
    args = arg_parser().parse_args()
    logger = _configure_logger(args.verbose, args.debug)

    if args.command == "check":
        handler = HandleCheck(arg_input=args.INPUT, logger=logger)

    if args.command == "config":
        handler = HandleConfig(arg_name=args.name, logger=logger)

    if args.command == "create":
        handler = HandleCreate(
            arg_input=args.input,
            arg_labels=args.labels,
            arg_name=args.name,
            arg_size=args.size,
            arg_testsplit=args.testsplit,
            arg_validsplit=args.validsplit,
            logger=logger,
        )

    if args.command == "predict":
        handler = HandlePredict(
            arg_model=args.model,
            arg_input=args.input,
            arg_output=args.output,
            arg_radius=args.radius,
            arg_shape=args.shape,
            logger=logger,
        )

    if args.command == "train":
        handler = HandleTrain(arg_config=args.config, arg_gpu=args.gpu, logger=logger)

    try:
        handler()
    except UnboundLocalError:
        logger.error(f"args.command defined as {args.command}. no handler defined")
