"""
UV-Vis Spectrum Analyzer for Plasmonic Nanoparticles

Usage:
    python spectrum_analyzer.py spectrum.txt
    python spectrum_analyzer.py
    python spectrum_analyzer.py spectrum.txt --smooth
"""

import os
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------
# Load spectrum
# ---------------------------------------------------------

def load_spectrum(filepath):

    data = np.loadtxt(filepath)

    if data.shape[1] != 2:
        raise ValueError("Input file must contain exactly two columns.")

    wavelength = data[:, 0]
    absorbance = data[:, 1]

    order = np.argsort(wavelength)

    return wavelength[order], absorbance[order]


# ---------------------------------------------------------
# FWHM
# ---------------------------------------------------------

def calculate_fwhm(wavelength, absorbance, peak_idx):

    half = absorbance[peak_idx] / 2

    left = peak_idx

    while left > 0 and absorbance[left] > half:
        left -= 1

    if left == 0:
        raise ValueError("Cannot calculate left side of FWHM.")

    x0, x1 = wavelength[left], wavelength[left + 1]
    y0, y1 = absorbance[left], absorbance[left + 1]

    left_wl = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    right = peak_idx

    while right < len(absorbance) - 1 and absorbance[right] > half:
        right += 1

    if right == len(absorbance) - 1:
        raise ValueError("Cannot calculate right side of FWHM.")

    x0, x1 = wavelength[right - 1], wavelength[right]
    y0, y1 = absorbance[right - 1], absorbance[right]

    right_wl = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    return right_wl - left_wl, left_wl, right_wl


# ---------------------------------------------------------
# Bipyramid analysis
# ---------------------------------------------------------

def analyze_bipyramid(wavelength, absorbance, smooth=False):

    if smooth:

        window = min(15, len(absorbance))

        if window % 2 == 0:
            window -= 1

        if window >= 5:
            absorbance = savgol_filter(absorbance, window_length=window, polyorder=3)

    prominence = np.max(absorbance) * 0.02

    peaks, properties = find_peaks(
        absorbance,
        prominence=prominence,
        distance=20
    )

    if len(peaks) < 2:
        raise ValueError(
            "Less than two significant peaks were detected.\n"
            "Try using --smooth."
        )

    # Two most prominent peaks
    top2 = peaks[np.argsort(properties["prominences"])[-2:]]

    # First peak = TSPR
    # Second peak = LSPR
    tspr_idx, lspr_idx = sorted(top2, key=lambda i: wavelength[i])

    tspr_wl = float(wavelength[tspr_idx])
    tspr_abs = float(absorbance[tspr_idx])

    lspr_wl = float(wavelength[lspr_idx])
    lspr_abs = float(absorbance[lspr_idx])

    abs400 = float(np.interp(400, wavelength, absorbance))

    fwhm, left, right = calculate_fwhm(
        wavelength,
        absorbance,
        lspr_idx
    )

    return {

        "LSPR wavelength (nm)": lspr_wl,
        "LSPR absorbance": lspr_abs,

        "TSPR wavelength (nm)": tspr_wl,
        "TSPR absorbance": tspr_abs,

        "LSPR - TSPR": lspr_abs - tspr_abs,

        "LSPR / TSPR": lspr_abs / tspr_abs,

        "Absorbance at 400 nm": abs400,

        "TSPR / Abs(400 nm)": tspr_abs / abs400,

        "LSPR FWHM (nm)": fwhm,
        "LSPR FWHM left (nm)": left,
        "LSPR FWHM right (nm)": right,
    }


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def analyze(filepath, shape="bipyramid", smooth=False):

    wavelength, absorbance = load_spectrum(filepath)

    if shape == "bipyramid":
        return analyze_bipyramid(
            wavelength,
            absorbance,
            smooth=smooth
        )

    raise NotImplementedError(f"Shape '{shape}' not implemented.")


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def write_report(result, input_file):

    output = os.path.splitext(input_file)[0] + "_analyzed.txt"

    with open(output, "w") as f:

        f.write("Spectrum Analysis Results\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Input file : {input_file}\n\n")

        for key, value in result.items():
            f.write(f"{key}: {value:.4f}\n")

    return output


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help="Spectrum file"
    )

    parser.add_argument(
        "--shape",
        default="bipyramid",
        help="Nanoparticle shape"
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Smooth spectrum before peak detection"
    )

    args = parser.parse_args()

    filepath = args.filepath

    if filepath is None:
        filepath = input("Spectrum file: ").strip()

    if not os.path.isfile(filepath):
        raise SystemExit(f"File not found: {filepath}")

    try:

        result = analyze(
            filepath,
            shape=args.shape,
            smooth=args.smooth
        )

        print("\nResults")
        print("-" * 30)

        for k, v in result.items():
            print(f"{k:<24}: {v:.4f}")

        output = write_report(result, filepath)

        print(f"\nSaved to: {output}")

    except Exception as e:

        print("\nAnalysis failed:")
        print(e)