import numpy as np
import pytest

from pypeit_wrapit.common import stack_spectra


def make_spectrum(
    lam_start=4000.0,
    lam_end=5000.0,
    npts=500,
    scale=1.0,
    noise=1.0,
    slope=0.0,
):
    lam = np.linspace(lam_start, lam_end, npts)
    flux = scale * (1.0 + slope * (lam - lam.mean()))
    std = np.full_like(flux, noise)
    return np.column_stack([lam, flux, std])


def test_single_spectrum_identity():
    """Single spectrum stack should reproduce binned spectrum."""
    sp = make_spectrum(scale=2.0, noise=0.1)

    bin_size = 5.0
    stacked = stack_spectra([sp], bin_size=bin_size)
    lam_s, flux_s, std_s = stacked.T

    # Bin original spectrum manually
    lam = sp[:, 0]
    flux = sp[:, 1]
    std = sp[:, 2]

    lam_min, lam_max = lam.min(), lam.max()
    num_bins = int(np.floor((lam_max - lam_min) / bin_size)) + 1
    bin_idx = ((lam - lam_min) // bin_size).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < num_bins)

    idx = bin_idx[valid]
    inv_var = std[valid] ** -2

    flux_ref = np.bincount(idx, weights=flux[valid] * inv_var, minlength=num_bins)
    ivar_sum = np.bincount(idx, weights=inv_var, minlength=num_bins)
    pos = ivar_sum > 0
    flux_ref[pos] = flux_ref[pos] / ivar_sum[pos]

    flux_ref = flux_ref[pos]

    assert np.allclose(flux_s, flux_ref, rtol=1e-6)
    assert np.all(std_s > 0)


def test_throughput_invariance_shape():
    """Stack should be invariant to multiplicative throughput differences."""
    sp1 = make_spectrum(scale=1.0, noise=1.0)
    sp2 = make_spectrum(scale=5.0, noise=5.0)  # same S/N, different throughput

    stacked = stack_spectra([sp1, sp2], bin_size=5.0)

    lam, flux, std = stacked.T

    # Shape should match intrinsic spectrum
    ref = sp1[:, 1] / np.mean(sp1[:, 1])
    test = flux / np.mean(flux)

    assert np.allclose(test, ref[: len(test)], rtol=1e-3)


def test_no_bright_spectrum_domination():
    """A high-throughput but noisy spectrum should not dominate."""
    sp_good = make_spectrum(scale=1.0, noise=0.5)
    sp_bad = make_spectrum(scale=10.0, noise=10.0)

    stacked = stack_spectra([sp_good, sp_bad], bin_size=5.0)

    lam, flux, std = stacked.T

    # Result should be close to the good spectrum
    ref = sp_good[:, 1] / np.mean(sp_good[:, 1])
    test = flux / np.mean(flux)

    assert np.allclose(test, ref[: len(test)], rtol=1e-2)


def test_inverse_variance_weighting():
    """Lower-noise spectrum should dominate the stack."""
    sp1 = make_spectrum(scale=1.0, noise=0.1)
    sp2 = make_spectrum(scale=1.0, noise=1.0)

    stacked = stack_spectra([sp1, sp2], bin_size=5.0)

    lam, flux, std = stacked.T

    # Expect uncertainty close to best spectrum
    assert np.nanmedian(std) < 0.2


def test_weighted_wavelength_centroid():
    """Centroids should be closer to the lower-variance spectrum."""
    sp1 = make_spectrum(lam_start=4000, lam_end=5000, noise=0.1)
    sp2 = make_spectrum(lam_start=4001, lam_end=5001, noise=1.0)

    bin_size = 10.0
    stacked = stack_spectra([sp1, sp2], bin_size=bin_size)
    lam = stacked[:, 0]

    lam_min_shared = max(sp1[:, 0].min(), sp2[:, 0].min())
    lam_max_shared = min(sp1[:, 0].max(), sp2[:, 0].max())

    num_bins = int(np.floor((lam_max_shared - lam_min_shared) / bin_size)) + 1

    def binned_centroids(sp):
        _lam = sp[:, 0]
        idx = ((_lam - lam_min_shared) // bin_size).astype(int)

        valid = (idx >= 0) & (idx < num_bins)

        lam_sum = np.bincount(
            idx[valid],
            weights=_lam[valid],
            minlength=num_bins,
        )
        count = np.bincount(
            idx[valid],
            minlength=num_bins,
        )

        out = np.full(num_bins, np.nan)
        m = count > 0
        out[m] = lam_sum[m] / count[m]

        return out

    lam1 = binned_centroids(sp1)
    lam2 = binned_centroids(sp2)

    bin_idx = ((lam - lam_min_shared) // bin_size).astype(int)
    mask = (bin_idx >= 0) & (bin_idx < num_bins)
    mask &= np.isfinite(lam1[bin_idx]) & np.isfinite(lam2[bin_idx])

    d1 = np.abs(lam[mask] - lam1[bin_idx[mask]])
    d2 = np.abs(lam[mask] - lam2[bin_idx[mask]])

    assert np.all(d1 < d2)


def test_no_overlap_raises():
    sp1 = make_spectrum(lam_start=4000, lam_end=4500)
    sp2 = make_spectrum(lam_start=4600, lam_end=5000)

    with pytest.raises(ValueError, match="No overlap"):
        stack_spectra([sp1, sp2])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        stack_spectra([])


def test_nan_handling():
    """NaNs in flux or std should not crash stacking."""
    sp1 = make_spectrum()
    sp2 = make_spectrum()

    sp2[100:120, 1] = np.nan
    sp2[200:210, 2] = np.nan

    stacked = stack_spectra([sp1, sp2], bin_size=5.0)

    lam, flux, std = stacked.T

    assert np.all(np.isfinite(flux))
    assert np.all(std > 0)
