import argparse
import logging
import os
from collections.abc import Sequence

from .config import ensure_dirs, load_config
from .manifest import discover_sources, load_specs


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assetpipe",
        description="Batch pipeline for isometric pixel-art assets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("source", "Generate source images from a JSONL manifest."),
        ("build", "Build assets from a directory of source PNGs."),
        ("all", "Generate sources and build all assets."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "input",
            help=(
                "Path to a JSONL manifest."
                if command in {"source", "all"}
                else "Directory containing source PNGs."
            ),
        )
        subparser.add_argument(
            "--config",
            default=os.environ.get("ASSETPIPE_CONFIG", "config.yaml"),
            help="YAML configuration path (default: config.yaml).",
        )
        subparser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate outputs even when they already exist.",
        )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    ensure_dirs(config)

    if args.command in {"source", "all"}:
        specs = load_specs(args.input)
        LOGGER.info("[%s] validated %d asset spec(s)", args.command, len(specs))
    elif args.command == "build":
        sources = discover_sources(args.input)
        LOGGER.info("[build] discovered %d source PNG(s)", len(sources))


if __name__ == "__main__":
    main()
