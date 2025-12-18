from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path
from typing import Sequence
import re
import shutil
import subprocess

from astropy.io import fits
import numpy as np

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
    if not spec_arrays:
        raise ValueError("`spec_arrays` is empty")

    lam_ranges = [(np.min(sp[:, 0]), np.max(sp[:, 0])) for sp in spec_arrays]
    lam_min = max(lo for lo, _ in lam_ranges)
    lam_max = min(hi for _, hi in lam_ranges)
    if lam_min >= lam_max:
        raise ValueError(
            f"No overlap of wavelength ranges between spectra (found {lam_min} to {lam_max})"
        )

    # Normalize spectra
    spec_norm = []
    norms = []
    norm_low = max(norm_lim_low, lam_min)
    norm_high = min(norm_lim_upp, lam_max)

    for sp in spec_arrays:
        sp_norm = np.array(sp, copy=True, dtype=float)
        mask = (sp_norm[:, 0] >= norm_low) & (sp_norm[:, 0] <= norm_high)
        if not np.any(mask):
            raise ValueError(
                "No data points within normalization limits; check input spectra and overlap region"
            )
        mean_val = np.nanmean(sp_norm[:, 1][mask])
        if not np.isfinite(mean_val) or mean_val == 0:
            raise ValueError(
                "Unable to compute normalization; check input spectra and overlap region"
            )
        sp_norm[:, 1:] /= mean_val
        norms.append(mean_val)
        spec_norm.append(sp_norm)

    num_bins = int(np.floor((lam_max - lam_min) / bin_size)) + 1

    lam_all = np.concatenate([sp[:, 0] for sp in spec_norm])
    flux_all = np.concatenate([sp[:, 1] for sp in spec_norm])
    std_all = np.concatenate([sp[:, 2] for sp in spec_norm])

    # Bin assignment (explicit masking, no clipping)
    bin_idx = ((lam_all - lam_min) // bin_size).astype(int)
    bin_valid = (bin_idx >= 0) & (bin_idx < num_bins)
    lam_all = lam_all[bin_valid]
    flux_all = flux_all[bin_valid]
    std_all = std_all[bin_valid]
    bin_idx = bin_idx[bin_valid]

    # Prepare output arrays
    lam_new = np.full(num_bins, np.nan, dtype=float)
    flux_new = np.full(num_bins, np.nan, dtype=float)
    std_new = np.full(num_bins, np.nan, dtype=float)

    # Inverse-variance weighted wavelength and flux
    # Shared valid mask
    ivar_valid = np.isfinite(flux_all) & np.isfinite(std_all) & (std_all > 0)
    idx = bin_idx[ivar_valid]
    inv_var = std_all[ivar_valid] ** -2
    # Shared inverse-variance sum
    ivar_sum = np.bincount(idx, weights=inv_var, minlength=num_bins)
    pos = ivar_sum > 0
    # Wavelength: inverse-variance weighted mean
    lam_weighted = np.bincount(
        idx, weights=lam_all[ivar_valid] * inv_var, minlength=num_bins
    )
    lam_new[pos] = lam_weighted[pos] / ivar_sum[pos]
    # Flux: inverse-variance weighted mean
    flux_weighted = np.bincount(
        idx, weights=flux_all[ivar_valid] * inv_var, minlength=num_bins
    )
    flux_new[pos] = flux_weighted[pos] / ivar_sum[pos]
    # Std: summed inverse variance
    std_new[pos] = np.sqrt(1.0 / ivar_sum[pos])

    # Drop bins without valid flux
    good = np.isfinite(lam_new) & np.isfinite(flux_new)
    lam_new = lam_new[good]
    flux_new = flux_new[good]
    std_new = std_new[good]

    # Restore flux scale using an inverse-variance weighted
    # throughput estimate
    a = np.asarray(norms)
    weights = []
    for sp_norm in spec_norm:
        mask = (sp_norm[:, 0] >= norm_low) & (sp_norm[:, 0] <= norm_high)
        if not np.any(mask):
            weights.append(0.0)
            continue
        weights.append(np.nansum(1.0 / sp_norm[:, 2][mask] ** 2))

    anchor_scale = np.average(a, weights=np.asarray(weights))

    flux_new *= anchor_scale
    std_new *= anchor_scale

    stacked = np.vstack([lam_new, flux_new, std_new]).T

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
