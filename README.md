# pypeit-wrapit

Lightweight wrappers to run **[PypeIt](https://pypeit.readthedocs.io/en/stable/)** reductions in a single step.

Supports the **Liverpool Telescope (LT) SPRAT** spectrograph for now.

`pypeit-wrapit` implements a small CLI (`pypeit-wrapit lt-sprat`) that prepares raw inputs, runs the relevant
`PypeIt` pipelines, performs flux calibration, optionally stacks spectra, and writes final ASCII spectral products.

---

## Installation

Install the repository via `pip`:
```bash
pip install git+https://github.com/Lyalpha/pypeit-wrapit.git
```

---

## Command Line Interface (CLI)

### Usage

The main CLI entry point is:

```bash
pypeit-wrapit
```
 You can append `-v` or `-vv` flags after this entrypoint (and before any of the below commands) for increased verbosity.

All available commands can be listed with:
```bash
pypeit-wrapit --help
```

#### LT SPRAT Command

```bash
pypeit-wrapit lt-sprat INPUT_FILES OUTPUT_DIR [OPTIONS]
```

### Arguments

- **INPUT_FILES**
  One or more input spectra FITS files to process.
  You may pass:
  - One or more file paths
  or
  - A glob pattern (shell-expanded)
  This should include the science frames and any necessary calibration frames (e.g. arcs)
  for **a single target and observational setup**.

- **OUTPUT_DIR**
  Output directory for intermediate and final reduced products.

---

### Options

- `--sensitivity-file`
  Path to sensitivity file.
  Default: packaged sensitivity files precomputed with `PypeIt` for SPRAT .

- `--stacked-bin-size`
  Bin size in angstroms for the stacked spectrum.
  Default: `5.0`

- `--lam-lim-low`, `--lam-lim-upp`
  Wavelength limits in angstroms for output spectra.
  Defaults: `4000.0`, `9000.0`

- `--overwrite`
  Removes contents of `OUTPUT_DIR` if it is not empty (interactive confirmation).

- `--cleanup`
  Removes intermediate `raw` and `pypeit_products` directories within `OUTPUT_DIR` after completion.

---

## Examples

### Run with a shell-expanded glob at INFO verbosity

```bash
pypeit-wrapit -v lt-sprat path/to/*.fits /tmp/outdir
```

### Pass a list of specific file paths and run with DEBUG verbosity

```bash
pypeit-wrapit -vv lt-sprat obj_spec1.fits obj_spec2.fits obj_arc.fits /tmp/outdir
```

### Specify a sensitivity file and request cleanup with WARNING verbosity

```bash
pypeit-wrapit lt-sprat data/*.fits /tmp/outdir \
  --sensitivity-file calibs/lt_sprat/sensfunc_red.fits \
  --cleanup
```

---

## Behavior Notes

### Flux Scaling

- Fluxes in ASCII outputs are return in units of `erg / s / cm^2 / Å`

### Stacking

- If there are multiple spectra of the target provided, they are:
  - Normalized by their mean flux within a broad normalization window
  - Resampled onto a regular wavelength grid
  - Stacked using inverse-variance weighting

The stacked spectrum has its observation time updated and output as a single stacked spectrum file
(see below).

### Output Files

Final ASCII spectra are written under:

```
OUTPUT_DIR/final/
```

#### Naming Convention

If a single spectrum of the target is passed:
  ```
  {TARGET}_LTSPRAT_{dispname}_{YYYY-MM-DDTHH:MM}.dat
  ```

If multiple spectra of the target are passed, individual spectra are written as:
    ```
    individual_{TARGET}_LTSPRAT_{dispname}_{YYYY-MM-DDTHH:MM}.dat
    ```

with their stacked spectrum written following the same convention as the single spectrum case.

- `{TARGET}` is the target name extracted from the FITS headers
- `{dispname}` is the spectrograph disperser name
- `{YYYY-MM-DDTHH:MM}` is the `DATE-OBS` timestamp of the observation

---

## Dependencies

- Python 3.12+
- pypeit (mpursiai's fork with LT SPRAT support)
- astropy
- numpy
- loguru
- rich-click

---

## Code Structure

Key files in the repository:

- **CLI entry point**
  `src/pypeit_wrapit/cli.py`

- **LT SPRAT workflow**
  `src/pypeit_wrapit/lt_sprat.py`

- **Common utilities** (flux writing, FITS unpacking, stacking)
  `src/pypeit_wrapit/common.py`
