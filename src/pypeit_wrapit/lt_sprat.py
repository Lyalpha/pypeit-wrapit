from datetime import datetime
from pathlib import Path
from typing import Sequence
import os
import shutil

from astropy.io import fits
from astropy.time import Time
from loguru import logger
import numpy as np

from pypeit_wrapit.common import (
    get_dispname,
    get_observation_date,
    get_spec1d_files,
    get_target_name,
    package_resource_path,
    run_pypeit,
    run_pypeit_flux_calib,
    run_pypeit_setup_one_config,
    stack_spectra,
    symlink_into_dir,
    unpack_spec1d_fits,
    write_flux_file,
)

SPECTROGRAPH = "lt_sprat"
SENSITIVITY_FILE_MAPPING = {"blue": "sensfunc_blue.fits", "red": "sensfunc_red.fits"}


def resource_path(filename: str) -> Path:
    return package_resource_path(os.path.join(SPECTROGRAPH, filename))


def run_object(
    input_files: Sequence[str | os.PathLike],
    output_dir: str | os.PathLike,
    sensitivity_file: str | os.PathLike | None = None,
    stacked_bin_size: float = 5.0,
    lam_lim_low: float = 4000.0,
    lam_lim_upp: float = 9000.0,
    cleanup: bool = False,
):
    input_paths = [Path(fp).absolute() for fp in input_files]
    for input_path in input_paths:
        if not input_path.is_file():
            if input_path.is_dir():
                raise IsADirectoryError(
                    f"Expected file(s) to process as input, but got a directory: {input_path}"
                )
            raise FileNotFoundError(f"Input file not found: {input_path}")
    output_dir_path = Path(output_dir).absolute()

    logger.info(
        f"Running LT SPRAT reduction for the files: {'\n'.join(map(str, input_paths))}"
    )
    logger.info(f"Output will be saved to: {output_dir_path}")

    raw_dir_path = output_dir_path / "raw"
    pypeit_dir_path = output_dir_path / "pypeit_products"
    final_dir_path = output_dir_path / "final"

    for dir_path in (raw_dir_path, pypeit_dir_path, final_dir_path):
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy the raw files to the raw_directory
    for input_path in input_paths:
        logger.debug(f"Linking file {input_path} to raw directory")
        symlink_into_dir(input_path, raw_dir_path)

    # LT data are CCD-reduced, but PypeIt still requires bias and flat files to be available
    # so we symlink the packaged archival files
    logger.debug(
        "Linking archival bias and flat files to the raw directory (unused for LT SPRAT)"
    )
    symlink_into_dir(resource_path("bias.fits"), raw_dir_path)
    symlink_into_dir(resource_path("flat.fits"), raw_dir_path)

    # Setup PypeIt
    logger.info("Setting up PypeIt reduction with command call to `pypeit_setup`")
    run_pypeit_setup_one_config(
        raw_dir=raw_dir_path,
        spectrograph=SPECTROGRAPH,
        output_path=pypeit_dir_path,
        cwd=pypeit_dir_path,
    )

    # Run PypeIt
    logger.info("Running PypeIt reduction with command call to `run_pypeit`")
    spectrograph_set = f"{SPECTROGRAPH}_A"
    pypeit_file = pypeit_dir_path / spectrograph_set / f"{spectrograph_set}.pypeit"
    run_pypeit(
        reduction_dir=pypeit_dir_path,
        pypeit_file=pypeit_file,
        cwd=pypeit_dir_path,
    )

    # Get the disperser name
    dispname = get_dispname(pypeit_file=pypeit_file, setup="A")
    logger.info(f"Identified disperser name as: {dispname}")

    if sensitivity_file is None:
        logger.info(
            "No sensitivity file provided. Using a packaged archival sensitivity file."
        )
        try:
            sensitivity_filename = SENSITIVITY_FILE_MAPPING[dispname]
        except KeyError:
            raise KeyError(
                f"No valid file mapping available for dispname='{dispname}'. Allowed values"
                f" are {list(SENSITIVITY_FILE_MAPPING.keys())}"
            )
        sensitivity_file_path = resource_path(sensitivity_filename)
        logger.debug(f"Using sensitivity file: {sensitivity_file_path}")
    else:
        sensitivity_file_path = Path(sensitivity_file).absolute()
        logger.debug(f"Using user-provided sensitivity file: {sensitivity_file_path}")

    spec1d_file_paths = get_spec1d_files(pypeit_dir_path)
    logger.info(f"Found {len(spec1d_file_paths)} spec1d files to be flux calibrated.")

    logger.info("Writing flux calibration parameter file.")
    flux_file_path = write_flux_file(
        reduction_dir=pypeit_dir_path,
        spec1d_files=spec1d_file_paths,
        sensfile=sensitivity_file_path,
    )

    # Flux calibrate the spectra via PypeIt
    logger.info("Running flux calibration with command call to `pypeit_flux_calib`")
    run_pypeit_flux_calib(
        flux_file=flux_file_path,
        cwd=pypeit_dir_path,
    )

    spec1d_filepaths_target_mapping: dict[str, list[Path]] = {}
    for spec1d_file_path in spec1d_file_paths:
        target_name = get_target_name(spec1d_file_path)
        spec1d_filepaths_target_mapping.setdefault(target_name, []).append(
            spec1d_file_path
        )

    for target_name, paths_for_target in spec1d_filepaths_target_mapping.items():
        n_spec = len(paths_for_target)
        logger.info(
            f"Unpacking flux-calibrated spectra ({n_spec}) for {target_name} and writing final ASCII file(s)."
        )
        do_stacking = n_spec > 1
        qualifier = "individual_" if do_stacking else ""

        date_obs_vals: list[datetime] = []
        spec1d_arrays: list[np.ndarray] = []
        stacked_hdr: fits.header.Header | None = None

        # Write individual spectra and optionally collect data for stacking
        for spec1d_file_path in paths_for_target:
            date_obs = get_observation_date(spec1d_file_path)
            date_obs_str = date_obs.strftime("%Y-%m-%dT%H:%M")

            write_filename = (
                f"{qualifier}{target_name}_LTSPRAT_{dispname}_{date_obs_str}.dat"
            )
            write_path = final_dir_path / write_filename

            spec1d_array, spec1d_hdr = unpack_spec1d_fits(
                spec1d_file=spec1d_file_path,
                lam_lim_low=lam_lim_low,
                lam_lim_upp=lam_lim_upp,
                write_ascii_to=write_path,
            )
            logger.debug(f"Wrote flux-calibrated spectrum to: {write_path}")

            if do_stacking:
                date_obs_vals.append(date_obs)
                spec1d_arrays.append(spec1d_array)
                if stacked_hdr is None:
                    stacked_hdr = spec1d_hdr

        # Stack spectra if applicable
        if do_stacking and spec1d_arrays and stacked_hdr is not None:
            logger.info(f"Stacking individual spectra for target {target_name}")

            mean_timestamp = sum(dt.timestamp() for dt in date_obs_vals) / len(
                date_obs_vals
            )
            mean_date = datetime.fromtimestamp(mean_timestamp)
            mean_date_str = mean_date.strftime("%Y-%m-%dT%H:%M")

            stacked_hdr["DATE-OBS"] = mean_date.isoformat()
            stacked_hdr["MJD"] = Time(mean_date).mjd

            stacked_write_path = (
                final_dir_path / f"{target_name}_LTSPRAT_{dispname}_{mean_date_str}.dat"
            )

            stacked_spectrum = stack_spectra(
                spec_arrays=spec1d_arrays,
                bin_size=stacked_bin_size,
                hdr=stacked_hdr,
                write_ascii_to=stacked_write_path,
            )
            lam_low = stacked_spectrum[0, 0]
            lam_upp = stacked_spectrum[-1, 0]
            logger.debug(
                f"Stacked spectrum wavelength range: {lam_low:.1f} - {lam_upp:.1f} Å"
            )
            logger.debug(
                f"Wrote stacked flux-calibrated spectrum to: {stacked_write_path}"
            )

    if cleanup:
        logger.info("Cleaning up intermediate directories.")
        shutil.rmtree(raw_dir_path)
        logger.debug(f"Removed raw directory: {raw_dir_path}")
        shutil.rmtree(pypeit_dir_path)
        logger.debug(f"Removed pypeit products directory: {pypeit_dir_path}")

    logger.info(
        f"LT SPRAT reduction complete. Final products are located in: {final_dir_path}"
    )
