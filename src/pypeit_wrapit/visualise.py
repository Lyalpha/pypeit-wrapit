from pathlib import Path
import sys

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # type: ignore[import]

plt.style.use(["science", "no-latex"])


def make_spectrum_fig(
    spectrum_file: str,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a sample spectrum figure for demonstration purposes.

    Returns:
        fig: The matplotlib Figure object.
        ax: The matplotlib Axes object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        data = np.loadtxt(spectrum_file)
    except Exception as e:
        logger.error(f"Failed to load spectrum file '{spectrum_file}': {e}")
        raise
    if data.ndim != 2 or data.shape[1] < 3:
        logger.error(
            "Spectrum file must have at least 3 columns: wavelength, flux, std."
        )
        sys.exit(1)

    lam = data[:, 0]
    flux = data[:, 1]
    std = data[:, 2]

    ax.fill_between(
        lam,
        flux - std,
        flux + std,
        color="C0",
        alpha=0.2,
        label="±1σ",
        linewidth=0,
    )
    ax.plot(lam, flux, color="C0", linewidth=1.5, label="Flux")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    if title is not None:
        ax.set_title(title)
    else:
        spectrum_filename = Path(spectrum_file).name
        ax.set_title(spectrum_filename)
    fig.tight_layout()

    return fig, ax


def show_spectrum(
    spectrum_file: str,
    title: str | None,
    output_path: str | None,
    no_show: bool = False,
) -> None:
    fig, _ = make_spectrum_fig(spectrum_file, title)
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if not no_show:
        plt.show()
