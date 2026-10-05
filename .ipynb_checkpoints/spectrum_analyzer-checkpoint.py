"""
UV-Vis Spectrum Analyzer for Plasmonic Nanoparticles
======================================================

Takes a wavelength (x) vs absorbance (y) spectrum and extracts the
characteristic peak values used to describe anisotropic nanoparticles.

Currently implemented shapes:
    - "bipyramid"

Planned shapes (raise NotImplementedError for now):
    - "cube"   (your "squares")
    - "sphere" (note: spheres usually only show ONE peak, not two -
                 the TSPR/LSPR logic will need to change for this shape)

Quick glossary (since chemistry isn't your background):
    TSPR  = Transverse Surface Plasmon Resonance -> the first, smaller peak
            (shorter wavelength).
    LSPR  = Longitudinal Surface Plasmon Resonance -> the second, bigger peak
            (longer wavelength).
    FWHM  = Full Width at Half Maximum -> how wide a peak is, measured at
            half of its height.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Load a two-column txt file: wavelength <whitespace/tab> absorbance
# e.g.:
#   396.4   0.11354
#   396.9   0.11392
# ----------------------------------------------------------------------
def load_spectrum_txt(filepath, wavelength_min=None, wavelength_max=None):
    """
    Loads a two-column wavelength/absorbance txt file (space or tab
    separated, no header row expected).

    wavelength_min / wavelength_max:
        Optional. If given, only the rows within that range are kept, so
        you do NOT need to pre-trim the file in another program first -
        this does it for you in one step.

        IMPORTANT: do not set wavelength_min as high as 400. The TSPR
        peak (first peak) is often BELOW 400 nm, so trimming there can
        delete the peak you need. Leave some margin below 400
        (e.g. wavelength_min=350) so both peaks stay in the data. The
        400 nm value is only used later as a single reference point, not
        as the start of your usable range.

    Returns: (wavelength_array, absorbance_array)
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"Expected 2 columns (wavelength, absorbance) in '{filepath}', "
            f"found {data.shape[1]}."
        )

    wavelength = data[:, 0]
    absorbance = data[:, 1]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    absorbance = absorbance[order]

    if wavelength_min is not None or wavelength_max is not None:
        lo = wavelength_min if wavelength_min is not None else wavelength.min()
        hi = wavelength_max if wavelength_max is not None else wavelength.max()
        mask = (wavelength >= lo) & (wavelength <= hi)
        wavelength = wavelength[mask]
        absorbance = absorbance[mask]

        if len(wavelength) == 0:
            raise ValueError(
                f"No data left after trimming to [{lo}, {hi}] nm. "
                f"Check your range against the actual file content."
            )

    return wavelength, absorbance


# ----------------------------------------------------------------------
# Helper: find the Full Width at Half Maximum of a single peak
# ----------------------------------------------------------------------
def _find_half_max_crossings(wavelength, absorbance, peak_idx):
    """
    Given the index of a peak, find the wavelengths on either side of it
    where the curve crosses half of the peak's height, and interpolate
    between data points for a more precise result.

    Returns: (fwhm, left_wavelength, right_wavelength)

    Note: if the peak sits right at the edge of your data (no room to walk
    left or right), the result will just use the edge point and may be
    slightly underestimated. Make sure your scan range fully covers the peak.
    """
    peak_abs = absorbance[peak_idx]
    half_max = peak_abs / 2.0
    n = len(absorbance)

    # Walk left from the peak until absorbance drops to/below half_max
    left_idx = peak_idx
    while left_idx > 0 and absorbance[left_idx] > half_max:
        left_idx -= 1
    if left_idx == peak_idx:
        left_wavelength = wavelength[left_idx]
    elif absorbance[left_idx] == half_max:
        left_wavelength = wavelength[left_idx]
    else:
        x0, x1 = wavelength[left_idx], wavelength[left_idx + 1]
        y0, y1 = absorbance[left_idx], absorbance[left_idx + 1]
        left_wavelength = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    # Walk right from the peak until absorbance drops to/below half_max
    right_idx = peak_idx
    while right_idx < n - 1 and absorbance[right_idx] > half_max:
        right_idx += 1
    if right_idx == peak_idx:
        right_wavelength = wavelength[right_idx]
    elif absorbance[right_idx] == half_max:
        right_wavelength = wavelength[right_idx]
    else:
        x0, x1 = wavelength[right_idx - 1], wavelength[right_idx]
        y0, y1 = absorbance[right_idx - 1], absorbance[right_idx]
        right_wavelength = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    return right_wavelength - left_wavelength, left_wavelength, right_wavelength


# ----------------------------------------------------------------------
# Main analyzer class - designed so new shapes are easy to add later
# ----------------------------------------------------------------------
class SpectrumAnalyzer:
    """
    Analyzes UV-Vis wavelength/absorbance spectra of plasmonic nanoparticles.

    Usage:
        analyzer = SpectrumAnalyzer(shape="bipyramid")
        result = analyzer.analyze(wavelength_array, absorbance_array)
    """

    def __init__(self, shape="bipyramid", smooth=False, smooth_window=15, smooth_poly=3):
        """
        shape         : "bipyramid" (only one implemented so far)
        smooth        : set True if your raw data is noisy and peak
                         detection is finding false peaks
        smooth_window : window size for smoothing (must be odd, bigger = smoother)
        smooth_poly   : polynomial order used for smoothing
        """
        self.shape = shape.lower()
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

    def analyze(self, wavelength, absorbance, peak_distance=20, peak_prominence=0.01):
        """
        peak_distance    : minimum separation (in data points) required
                            between TSPR and LSPR peaks. Increase this if
                            the algorithm picks two peaks that are too close
                            together / are really the same bump.
        peak_prominence   : minimum "height above surroundings" a peak needs
                            to count. Lower this if a real peak is being
                            missed; raise it if noise is being picked up as
                            a fake peak.
        """
        wavelength = np.asarray(wavelength, dtype=float)
        absorbance = np.asarray(absorbance, dtype=float)

        if wavelength.shape != absorbance.shape:
            raise ValueError("wavelength and absorbance must be the same length.")

        # Make sure wavelength is sorted ascending (some instruments export it descending)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        absorbance = absorbance[order]

        if self.smooth:
            window = self.smooth_window
            if window % 2 == 0:
                window += 1
            window = min(window, len(absorbance) - 1 if len(absorbance) % 2 == 0 else len(absorbance))
            if window >= 5:
                absorbance = savgol_filter(absorbance, window_length=window, polyorder=self.smooth_poly)

        if self.shape == "bipyramid":
            return self._analyze_bipyramid(wavelength, absorbance, peak_distance, peak_prominence)
        elif self.shape == "cube":
            raise NotImplementedError("Analysis for cubes ('squares') is not implemented yet.")
        elif self.shape == "sphere":
            raise NotImplementedError(
                "Analysis for spheres is not implemented yet. "
                "Note: spheres typically show only ONE peak, so this will "
                "need different logic than TSPR/LSPR."
            )
        else:
            raise ValueError(f"Unknown shape '{self.shape}'.")

    # ------------------------------------------------------------------
    # Bipyramid-specific logic
    # ------------------------------------------------------------------
    def _analyze_bipyramid(self, wavelength, absorbance, peak_distance, peak_prominence):
        peaks, props = find_peaks(absorbance, distance=peak_distance, prominence=peak_prominence)

        if len(peaks) < 2:
            raise ValueError(
                "Could not find two distinct peaks (TSPR + LSPR) in this spectrum. "
                "Try lowering 'peak_prominence' or 'peak_distance', or turn on smoothing."
            )

        # Keep the two most prominent peaks, then order them by wavelength:
        # the first one (shortest wavelength) is TSPR, the second is LSPR.
        prominences = props["prominences"]
        top2 = peaks[np.argsort(prominences)[-2:]]
        tspr_idx, lspr_idx = sorted(top2, key=lambda i: wavelength[i])

        tspr_wavelength = float(wavelength[tspr_idx])
        tspr_abs = float(absorbance[tspr_idx])
        lspr_wavelength = float(wavelength[lspr_idx])
        lspr_abs = float(absorbance[lspr_idx])

        abs_at_400 = float(np.interp(400, wavelength, absorbance))

        fwhm, fwhm_left, fwhm_right = _find_half_max_crossings(wavelength, absorbance, lspr_idx)

        return {
            "shape": "bipyramid",
            "LSPR_wavelength_nm": lspr_wavelength,
            "LSPR_absorbance": lspr_abs,
            "TSPR_wavelength_nm": tspr_wavelength,
            "TSPR_absorbance": tspr_abs,
            "LSPR_minus_TSPR_absorbance": lspr_abs - tspr_abs,
            "LSPR_over_TSPR_ratio": lspr_abs / tspr_abs,
            "absorbance_at_400nm": abs_at_400,
            "TSPR_over_400_ratio": tspr_abs / abs_at_400,
            "LSPR_FWHM_nm": fwhm,
            "LSPR_FWHM_left_nm": fwhm_left,
            "LSPR_FWHM_right_nm": fwhm_right,
        }


# ----------------------------------------------------------------------
# Optional: visualize the spectrum with the detected peaks marked
# ----------------------------------------------------------------------
def plot_result(wavelength, absorbance, result, save_path="spectrum_analysis.png"):
    wavelength = np.asarray(wavelength, dtype=float)
    absorbance = np.asarray(absorbance, dtype=float)
    order = np.argsort(wavelength)
    wavelength, absorbance = wavelength[order], absorbance[order]

    plt.figure(figsize=(8, 5))
    plt.plot(wavelength, absorbance, label="Spectrum", color="steelblue")
    plt.scatter([result["TSPR_wavelength_nm"]], [result["TSPR_absorbance"]],
                color="orange", zorder=5, label="TSPR (1st peak)")
    plt.scatter([result["LSPR_wavelength_nm"]], [result["LSPR_absorbance"]],
                color="red", zorder=5, label="LSPR (2nd peak)")
    plt.hlines(result["LSPR_absorbance"] / 2,
               result["LSPR_FWHM_left_nm"], result["LSPR_FWHM_right_nm"],
               color="green", linestyle="--", label="LSPR FWHM")
    plt.axvline(400, color="gray", linestyle=":", label="400 nm reference")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"{result['shape'].capitalize()} spectrum analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- EXAMPLE 1: synthetic data, just so this file runs out of the box ---
    wl = np.linspace(300, 900, 600)
    abs_synthetic = (
        0.3 * np.exp(-((wl - 380) ** 2) / (2 * 20 ** 2))   # small TSPR-like bump
        + 0.9 * np.exp(-((wl - 650) ** 2) / (2 * 40 ** 2))  # bigger LSPR-like bump
        + 0.02 * np.random.normal(size=wl.shape)            # a bit of noise
    )

    analyzer = SpectrumAnalyzer(shape="bipyramid", smooth=True)
    result = analyzer.analyze(wl, abs_synthetic)

    print("Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    plot_result(wl, abs_synthetic, result)

    # --- EXAMPLE 2: loading your own data from a real txt file ---
    # Your files look like:
    #   396.4   0.11354
    #   396.9   0.11392
    # Just point this at your file. wavelength_min/max trims the range
    # for you (no need to pre-process it elsewhere). Start the range
    # well below 400 so the TSPR peak doesn't get cut off.
    #
    # wl, absb = load_spectrum_txt(
    #     "my_spectrum.txt",
    #     wavelength_min=350,
    #     wavelength_max=1100,
    # )
    # analyzer = SpectrumAnalyzer(shape="bipyramid")
    # result = analyzer.analyze(wl, absb)
    # print(result)
    # plot_result(wl, absb, result)
