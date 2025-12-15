import shutil
import subprocess
import re
from datetime import datetime
from importlib.resources import files, as_file
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits


PACKAGE = "pypeit_wrapit"


def package_resource_path(filename: str) -> Path:
    """Return a filesystem path for a packaged resource."""
    resource = files(PACKAGE) / "data" / filename
    with as_file(resource) as path:
        return Path(path).absolute()


def symlink_into_dir(src: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    """
    Create a symlink to `src` inside `dest_dir` using `src.name`.
    Returns the Path of the created symlink.
    Raises FileNotFoundError if destination exists and overwrite is False.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    link = dest_dir / src.name

    if link.exists() or link.is_symlink():
        # If already the correct symlink, nothing to do
        try:
            if link.is_symlink() and link.resolve() == src.resolve():
                return link
        except FileNotFoundError:
            # broken symlink, continue
            pass

        if overwrite:
            link.unlink()
        else:
            raise FileExistsError(f"`{link}` already exists")

    link.symlink_to(src)
    return link


def get_dispname(pypeit_file: Path, setup: str = "A") -> str:
    """
    Return the `dispname` for `Setup <setup>:` in the given .pypeit file <pypeit_file>.
    Raises FileNotFoundError if file missing and ValueError if setup or dispname not found.
    """
    p = Path(pypeit_file)
    if not p.exists():
        raise FileNotFoundError(f"`{p}` not found")

    content = p.read_text(encoding="utf-8").splitlines()
    setup_re = re.compile(r"^\s*Setup\s+(\S+)\s*:", re.IGNORECASE)
    disp_re = re.compile(r"^\s*dispname\s*:\s*(\S+)", re.IGNORECASE)
    target = setup

    in_block = False
    for line in content:
        m = setup_re.match(line)
        if m:
            name = m.group(1)
            in_block = name == target
            continue
        if in_block:
            m2 = disp_re.match(line)
            if m2:
                return m2.group(1).strip()
            # end block on encountering another "Setup ..." or "setup end"
            if re.match(r"^\s*Setup\s+\S+\s*:", line, re.IGNORECASE) or re.match(
                r"^\s*setup\s+end", line, re.IGNORECASE
            ):
                in_block = False

    raise ValueError(f"`dispname` for setup `{setup}` not found in `{p}`")


def get_header_keyword(fits_file: Path, key: str, ext: int | str = 0) -> str:
    """
    Return the value of the given header <key> from the spec1d <fits_file>.
    Raises FileNotFoundError if file missing and ValueError if keyword not found.
    """
    hdr = fits.getheader(fits_file, ext=ext)
    try:
        value = hdr[key]
    except KeyError:
        raise ValueError(f"`{key}` not found in header (ext={ext}) of `{fits_file}`")
    return value


def get_observation_date(
    fits_file: Path, key: str = "DATE-OBS", ext: int | str = 0
) -> datetime:
    """
    Return the observation date from the header of the given spec1d <fits_file> as a datetime object.
    The header keyword to read is given by <key> (default: "DATE-OBS").
    """
    date_obs_str = get_header_keyword(fits_file, key, ext=ext)
    return datetime.fromisoformat(date_obs_str)


def get_target_name(fits_file: Path, key: str = "TARGET", ext: int | str = 0) -> str:
    """
    Return the target name from the header of the given spec1d <fits_file>.
    The header keyword to read is given by <key> (default: "OBJECT").
    """
    target_name = get_header_keyword(fits_file, key, ext=ext)
    return target_name.strip()


def get_spec1d_files(reduction_dir: Path) -> list[Path]:
    """
    Return sorted `Path` objects for files matching `Science/spec1d_*.fits`
    under <reduction_dir>.
    """
    sci_dir = reduction_dir / "Science"
    if not sci_dir.exists():
        return []

    spec1d_files = sci_dir.glob("spec1d_*.fits")
    return sorted(spec1d_files, key=lambda p: p.name)


def write_flux_file(
    reduction_dir: Path, spec1d_files: list[Path], sensfile: Path
) -> Path:
    """
    Write a flux calibration parameter file `flux_file.txt` in <reduction_dir>, used by `pypeit_flux_calib`.
    The file lists the <spec1d_files> to be flux calibrated using the given <sensfile>.
    Returns the Path to the written file.
    """
    if not spec1d_files:
        raise ValueError("`spec1d_files` is empty")

    width = max(len(str(p)) for p in spec1d_files)

    out_path = reduction_dir / "flux_file.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("[fluxcalib] \n")
        f.write("    extrap_sens = True \n")
        f.write("flux read \n")
        f.write(f"    {'filename':{width}} | sensfile \n")

        for i, p in enumerate(spec1d_files):
            if i == 0:
                f.write(f"    {str(p):{width}} | {sensfile} \n")
            else:
                f.write(f"    {str(p):{width}} | \n")

        f.write("flux end \n")

    return out_path


def unpack_spec1d_fits(
    spec1d_file: Path,
    lam_lim_low: float | None = None,
    lam_lim_upp: float | None = None,
    write_ascii_to: Path | None = None,
) -> tuple[np.ndarray, fits.header.Header]:
    """
    Returns a Nx3 array with columns: wavelength, flux, sigma and the header of the reduced and
    flux calibrated spectrum in <spec1d_file>.
    """
    from pypeit.specobjs import SpecObjs

    spec_obj = SpecObjs.from_fitsfile(spec1d_file)
    wavelength, flux, ivar, _, _, _, hdr = spec_obj.unpack_object(ret_flam=True)

    std = np.full_like(ivar, np.inf, dtype=float)
    pos = ivar > 0
    std[pos] = np.sqrt(1.0 / ivar[pos])

    if (lam_lim_low, lam_lim_upp) != (None, None):
        wavelength_mask = np.ones_like(wavelength, dtype=bool)
        if lam_lim_low is not None:
            wavelength_mask &= wavelength >= lam_lim_low
        if lam_lim_upp is not None:
            wavelength_mask &= wavelength <= lam_lim_upp

        wavelength = wavelength[wavelength_mask]
        # PypeIt fluxes are in units of 1e-17 erg/s/cm2/Ang
        # https://pypeit.readthedocs.io/en/latest/fluxing.html
        flux = flux[wavelength_mask] * 1e-17  # convert to erg/s/cm2/Ang
        std = std[wavelength_mask] * 1e-17  # convert to erg/s/cm2/Ang

        if wavelength.size == 0:
            raise ValueError("No data within the requested wavelength limits")

    spec_array = np.column_stack((wavelength, flux, std))

    if write_ascii_to is not None:
        write_path = Path(write_ascii_to)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        hdr_lines = []
        for card in hdr.cards:
            if card.keyword not in ("HISTORY", "COMMENT", ""):
                hdr_lines.append(
                    f"{card.keyword} = {card.value} / {card.comment}".rstrip(" / ")
                )
        hdr_str = "\n".join(hdr_lines)
        hdr_str += "\n==== END OF FITS HEADER ===="
        hdr_str += "\nLambda    Flux        Std"
        np.savetxt(
            write_path,
            spec_array,
            fmt="%10.3f %11.4e %11.4e",
            header=hdr_str,
        )

    return spec_array, hdr


def stack_spectra(
    spec_arrays: Sequence[np.ndarray],
    bin_size: float = 5.0,
    norm_lim_low: float = 1000.0,
    norm_lim_upp: float = 25000.0,
    write_ascii_to: Path | None = None,
) -> np.ndarray:
    """
    Stack a sequence of spectra arrays (each with 3 columns: lambda, flux, std) onto a regular grid.

    Returns an (M ,3)  array with columns (wavelength, flux, std).
    """

    lam_min = float(np.max([sp[0][0] for sp in spec_arrays]))
    lam_max = float(np.min([sp[0][-1] for sp in spec_arrays]))
    if lam_min >= lam_max:
        raise ValueError("No overlap between spectra within `norm_lims`")

    # normalise each spectrum by mean flux in norm_lims region, further cropped by wavel_min/max
    norms = []
    for sp in spec_arrays:
        mask = (sp[0] > max(norm_lim_low, lam_min)) & (
            sp[0] < min(norm_lim_upp, lam_max)
        )
        if not np.any(mask):
            raise ValueError(
                "No data points within normalization limits; check input spectra and overlap region"
            )
        vals = sp[1][mask]
        mean_val = np.nanmean(vals)
        if not np.isfinite(mean_val) or mean_val == 0:
            raise ValueError(
                "Unable to compute normalization; check input spectra and overlap region"
            )
        norms.append(mean_val)
        # normalise the flux and sigma columns
        sp[1] = sp[1] / mean_val
        sp[2] = sp[2] / mean_val

    regular_grid = np.arange(lam_min, lam_max + bin_size, bin_size)

    # accumulate per-bin lists
    lam_bins = [[] for _ in range(len(regular_grid))]
    flux_bins = [[] for _ in range(len(regular_grid))]
    std_bins = [[] for _ in range(len(regular_grid))]

    for sp in spec_arrays:
        inds = np.digitize(sp[0], bins=regular_grid)
        for bin_idx in range(1, len(regular_grid) + 1):
            sel = inds == bin_idx
            if not np.any(sel):
                continue
            lam_bins[bin_idx - 1].extend(sp[0][sel].tolist())
            flux_bins[bin_idx - 1].extend(sp[1][sel].tolist())
            std_bins[bin_idx - 1].extend(sp[2][sel].tolist())

    # compute weighted means
    lam_new = np.full(len(regular_grid), np.nan, dtype=float)
    flux_new = np.full(len(regular_grid), np.nan, dtype=float)
    std_new = np.full(len(regular_grid), np.nan, dtype=float)

    for i in range(len(regular_grid)):
        lam_temp = np.asarray(lam_bins[i])
        flux_temp = np.asarray(flux_bins[i])
        std_temp = np.asarray(std_bins[i])

        if lam_temp.size == 0:
            continue

        lam_new[i] = np.nanmean(lam_temp)

        # filter finite and positive std
        valid = np.isfinite(flux_temp) & np.isfinite(std_temp) & (std_temp > 0)
        if not np.any(valid):
            continue

        f = flux_temp[valid]
        s = std_temp[valid]
        inv_var = 1.0 / (s * s)
        sum_inv_var = np.sum(inv_var)
        flux_new[i] = np.sum(f * inv_var) / sum_inv_var
        std_new[i] = np.sqrt(1.0 / sum_inv_var)

    # drop empty bins
    good = np.isfinite(lam_new)
    lam_new = lam_new[good]
    flux_new = flux_new[good]
    std_new = std_new[good]

    # restore average scaling
    mean_norm = np.nanmean(norms)
    flux_new = flux_new * mean_norm
    std_new = std_new * mean_norm

    stacked = np.vstack([lam_new, flux_new, std_new])

    if write_ascii_to is not None:
        write_path = Path(write_ascii_to)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(write_path),
            stacked,
            fmt="%10.3f %11.4e %11.4e",
            delimiter="\t",
            header="Lambda    Flux        Std",
        )

    return stacked


def run_command(
    cmd_name: str,
    args: Sequence[str],
    cwd: Path | None = None,
    timeout: int | None = 600,
) -> subprocess.CompletedProcess:
    """
    Run an external command and return CompletedProcess.
    Raises FileNotFoundError if the command is not on PATH and
    subprocess.CalledProcessError if the command returns a non-zero exit code.
    """
    cmd = [cmd_name, *list(args)]

    if shutil.which(cmd_name) is None:
        raise FileNotFoundError(f"{cmd_name} not found on PATH")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )

    return proc


def run_pypeit_setup_one_config(
    raw_dir: Path,
    spectrograph: str,
    extension: str = ".fits",
    output_path: Path | None = None,
    cwd: Path | None = None,
    timeout: int | None = 300,
) -> subprocess.CompletedProcess:
    """
    Wrapper to call `pypeit_setup` and run only a single configuration (A)
    """
    args = [
        "-r",
        str(raw_dir),
        "-s",
        spectrograph,
        "-e",
        extension,
        "-c",
        "A",
        "-o",
    ]
    if output_path is not None:
        args.extend(["-d", str(output_path)])
    return run_command("pypeit_setup", args, cwd=cwd, timeout=timeout)


def run_pypeit(
    reduction_dir: Path,
    pypeit_file: Path,
    cwd: Path | None = None,
    timeout: int | None = 900,
) -> subprocess.CompletedProcess:
    """
    Wrapper to call `run_pypeit`
    """
    args = ["-r", str(reduction_dir), str(pypeit_file)]
    return run_command("run_pypeit", args, cwd=cwd, timeout=timeout)


def run_pypeit_flux_calib(
    flux_file: Path,
    cwd: Path | None = None,
    timeout: int | None = 600,
) -> subprocess.CompletedProcess:
    """
    Wrapper to call `pypeit_flux_calib`
    """
    args = [str(flux_file)]
    return run_command("pypeit_flux_calib", args, cwd=cwd, timeout=timeout)
