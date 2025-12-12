# python
import glob
import sys
from pathlib import Path
from typing import Sequence, List, Set

import rich_click as click
from loguru import logger


def _collect_input_files(input_files: str) -> List[str]:
    """Expand comma-separated or glob pattern input files into a list of file paths."""

    if "," in input_files:
        inputs = [item.strip() for item in input_files.split(",") if item.strip()]
        return sorted(inputs)

    globbed_files: Set[str] = set()
    if any(ch in input_files for ch in ("*", "?", "[")):
        for match in glob.glob(input_files):
            if Path(match).is_file():
                globbed_files.add(match)
    return sorted(globbed_files)


def _setup_logging(verbose: int) -> None:
    """Configure loguru sink to stderr based on verbosity."""
    level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
    )


@click.group(context_settings={"show_default": True})
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase logging verbosity level; -v for INFO, -vv for DEBUG.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    """pypeit-wrapit CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command("lt-sprat")
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    help="One or more input spectra fits files to process. Pass either a list of file paths"
    " or a glob pattern. Glob patterns are expanded by your shell before being passed"
    " to this command.",
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, writable=True),
    required=True,
    help="Output directory for intermediate and final reduced products.",
)
@click.option(
    "--sensitivity-file",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, exists=True),
    default=None,
    help="Optional sensitivity file to use; defaults to packaged file based on disperser.",
)
@click.option(
    "--stacked-bin-size",
    type=float,
    default=5.0,
    help="Bin size in angstroms for stacked spectrum.",
)
@click.option(
    "--lam-lim-low",
    type=float,
    default=4000.0,
    help="Lower wavelength limit for output spectra.",
)
@click.option(
    "--lam-lim-upp",
    type=float,
    default=9000.0,
    help="Upper wavelength limit for output spectra.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing data in output directory if it exists. If the directory"
    "exists, is not empty, and this flag is not set, the command will abort.",
)
@click.option(
    "--cleanup",
    is_flag=True,
    default=False,
    help="Remove intermediate directories after completion.",
)
@click.pass_context
def run_lt_sprat(
    ctx: click.Context,
    input_files: Sequence[str],
    output_dir: str,
    sensitivity_file: str | None,
    stacked_bin_size: float,
    lam_lim_low: float,
    lam_lim_upp: float,
    overwrite: bool,
    cleanup: bool,
) -> None:
    """Run LT SPRAT reduction on input files for a single target and spectrograph set up."""

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            logger.error(
                f"Output directory {output_path} exists and is not empty. "
                "Use --overwrite to confirm removal of contents."
            )
            sys.exit(1)
        logger.warning(
            f"Output directory {output_path} exists and is not empty. "
            "Confirm removal of contents."
        )
        if click.confirm(
            f"Are you sure you want to overwrite the contents of {output_path}?"
        ):
            for item in output_path.iterdir():
                if item.is_dir():
                    import shutil

                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info(f"Cleared contents of {output_path}.")
        else:
            logger.info("Aborting operation.")
            sys.exit(0)

    from pypeit_wrapit.lt_sprat import run_object

    run_object(
        input_files=input_files,
        output_dir=output_dir,
        sensitivity_file=sensitivity_file,
        stacked_bin_size=stacked_bin_size,
        lam_lim_low=lam_lim_low,
        lam_lim_upp=lam_lim_upp,
        cleanup=cleanup,
    )


if __name__ == "__main__":
    cli()
