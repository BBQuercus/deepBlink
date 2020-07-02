"""Module that contains the command line app.

NOTE - Currently not functional. Will be populated in the next release.
Used for inferencing of new data with pretrained models.

Usage:
    deepblink <model> <input> [-o, --output] <output> [-t, --type] <type> [--verbose]
    deepblink -i
    deepblink --help

Examples:
    deepblink ./model.h5 ./data/ -o ./output/ -t csv

Arguments:
    <model>         Model .h5 file location.
    <input>         Input file/folder location.

Options:
    -i              Interactive mode.
    -o, --output    Output file/folder location. [default: input location]
    -t, --type      Output file type. [options: csv, txt] [default: csv]
    --verbose       Set program output to verbose. [default: quiet]
    -h, --help      Show this help screen.
    -V, --version   Show version.

Why does this file exist, and why not put this in __main__?
- When you run `python -mdeepblink` or `deepblink` directly, python will
    execute ``__main__.py`` as a script. That means there won't be any
    ``deepblink.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
    there's no ``deepblink.__main__`` in ``sys.modules``.
- Therefore, to avoid double excecution of the code, this split-up way is safer.
"""
import argparse

parser = argparse.ArgumentParser(description="Command description.")
parser.add_argument(
    "names", metavar="NAME", nargs=argparse.ZERO_OR_MORE, help="A name of something."
)


def main(args=None):
    """Addition of useful docstring.

    A very important change that CI checking triggers.

    Args:
        args: Something I eat.

    Returns:
        I'm free, I don't return anything.
    """
    args = parser.parse_args(args=args)
    print(args.names)
