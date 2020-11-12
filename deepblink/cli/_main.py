"""Main / entrypoint function for deepblinks CLI."""

import argparse
import os

from ._argparse import _parse_args_check
from ._argparse import _parse_args_config
from ._argparse import _parse_args_create
from ._argparse import _parse_args_download
from ._argparse import _parse_args_predict
from ._argparse import _parse_args_train
from ._check import HandleCheck
from ._config import HandleConfig
from ._create import HandleCreate
from ._download import HandleDownload
from ._logger import _configure_logger
from ._parseutil import Color
from ._parseutil import CustomFormatter
from ._parseutil import _add_utils
from ._predict import HandlePredict
from ._train import HandleTrain

# Removes tensorflow's information on CPU / GPU availablity.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def arg_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description=f"{Color.title}deepBlink's CLI \U0001F469\U0000200D\U0001F4BB to ferret out all of your spots.{Color.end}",
        epilog=(
            'See "deepblink <command> --help" to read about the specific subcommand.\n'
            'See the online wiki at "https://github.com/BBQuercus/deepBlink/wiki" for an overview.\n'
            "We hope you enjoy using deepBlink \U0001F603"
        ),
        formatter_class=CustomFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(
        title=f"{Color.optional}Commands for various situations{Color.end}",
        dest="command",
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    _parse_args_check(subparsers, parent_parser)
    _parse_args_config(subparsers, parent_parser)
    _parse_args_create(subparsers, parent_parser)
    _parse_args_download(subparsers, parent_parser)
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

    if args.command == "download":
        handler = HandleDownload(
            arg_input=args.input, arg_list=args.list, arg_all=args.all, logger=logger
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
