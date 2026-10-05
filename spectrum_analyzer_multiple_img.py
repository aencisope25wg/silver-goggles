"""
UV-Vis Spectrum Analyzer for Plasmonic Nanoparticles

Single-spectrum txt:
    python spectrum_analyzer.py spectrum.txt
    python spectrum_analyzer.py spectrum.txt --smooth

Multi-spectrum CSV  (wavelength col + one absorbance col per spectrum):
    python spectrum_analyzer.py spectra.csv
    python spectrum_analyzer.py spectra.csv --smooth

If no file is given the script will ask for one.
"""

import os
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_spectrum(filepath):
    """Two-column txt: wavelength  absorbance"""
    data = np.loadtxt(filepath)
    if data.ndim == 1 or data.shape[1] != 2:
        raise ValueError("Single-spectrum file must have exactly 2 columns.")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def load_csv_multi(filepath):
    """
    Multi-spectrum CSV.
    Row 0   : header  →  wavelength_label, spectrum_1_label, spectrum_2_label, ...
    Row 1.. : wavelength, abs_spectrum_1, abs_spectrum_2, ...
    Returns:
        wavelength  shape (W,)
        spectra     shape (W, N)   one column per spectrum
        labels      list of N strings (spectrum names from header)
    """
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")
    labels = header[1:]                                 # e.g. ["1","2",...,"110"]
    data   = np.loadtxt(filepath, delimiter=",", skiprows=1)
    wavelength = data[:, 0]
    spectra    = data[:, 1:]                            # (W, N)
    order      = np.argsort(wavelength)
    return wavelength[order], spectra[order], labels


# ---------------------------------------------------------
# Smoothing
# ---------------------------------------------------------

def smooth_spectrum(absorbance):
    window = min(15, len(absorbance))
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        return savgol_filter(absorbance, window_length=window, polyorder=3)
    return absorbance


# ---------------------------------------------------------
# FWHM
# ---------------------------------------------------------

def calculate_fwhm(wavelength, absorbance, peak_idx):
    half = absorbance[peak_idx] / 2.0

    left = peak_idx
    while left > 0 and absorbance[left] > half:
        left -= 1
    if left == 0:
        raise ValueError("Left FWHM boundary not found.")
    x0, x1 = wavelength[left],     wavelength[left + 1]
    y0, y1 = absorbance[left],     absorbance[left + 1]
    left_wl = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    right = peak_idx
    while right < len(absorbance) - 1 and absorbance[right] > half:
        right += 1
    if right == len(absorbance) - 1:
        raise ValueError("Right FWHM boundary not found.")
    x0, x1 = wavelength[right - 1], wavelength[right]
    y0, y1 = absorbance[right - 1], absorbance[right]
    right_wl = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    return right_wl - left_wl, left_wl, right_wl


# ---------------------------------------------------------
# Peak detection (shared)
# ---------------------------------------------------------

def detect_peaks(wavelength, absorbance, smooth=False):
    """
    Returns (peak_indices, properties, absorbance_used).
    prominence threshold = 2 % of the spectrum maximum (min 0.01).
    """
    abs_work = absorbance.copy()
    if smooth:
        abs_work = smooth_spectrum(abs_work)
    prominence = max(float(np.max(abs_work)) * 0.02, 0.01)
    peaks, props = find_peaks(abs_work, prominence=prominence, distance=20)
    return peaks, props, abs_work


# ---------------------------------------------------------
# Single-spectrum bipyramid analysis
# ---------------------------------------------------------

def analyze_bipyramid(wavelength, absorbance, smooth=False):
    peaks, props, abs_work = detect_peaks(wavelength, absorbance, smooth)

    if len(peaks) < 2:
        raise ValueError(
            "Fewer than two significant peaks detected.\n"
            "Try --smooth, or check the wavelength range covers both peaks."
        )

    top2             = peaks[np.argsort(props["prominences"])[-2:]]
    tspr_idx, lspr_idx = sorted(top2, key=lambda i: wavelength[i])

    tspr_wl  = float(wavelength[tspr_idx])
    tspr_abs = float(abs_work[tspr_idx])
    lspr_wl  = float(wavelength[lspr_idx])
    lspr_abs = float(abs_work[lspr_idx])
    abs400   = float(np.interp(400, wavelength, abs_work))

    fwhm, left, right = calculate_fwhm(wavelength, abs_work, lspr_idx)

    return {
        "LSPR wavelength (nm)": lspr_wl,
        "LSPR absorbance":      lspr_abs,
        "TSPR wavelength (nm)": tspr_wl,
        "TSPR absorbance":      tspr_abs,
        "LSPR - TSPR":          lspr_abs - tspr_abs,
        "LSPR / TSPR":          lspr_abs / tspr_abs,
        "Absorbance at 400 nm": abs400,
        "TSPR / Abs(400 nm)":   tspr_abs / abs400,
        "LSPR FWHM (nm)":       fwhm,
        "LSPR FWHM left (nm)":  left,
        "LSPR FWHM right (nm)": right,
    }


# ---------------------------------------------------------
# Per-spectrum analysis used in the multi-spectrum loop
# ---------------------------------------------------------

def analyze_one(wavelength, absorbance, smooth=False):
    """
    Returns a dict:
        n_peaks  int
        status   str   "no peaks" | "1 peak" | "2 peaks" | "N peaks (top 2 used)"
        result   dict  of measured values, or None
    """
    peaks, props, abs_work = detect_peaks(wavelength, absorbance, smooth)
    n = len(peaks)

    if n == 0:
        return {"n_peaks": 0, "status": "no peaks", "result": None}

    if n == 1:
        pk = peaks[0]
        return {
            "n_peaks": 1,
            "status": "1 peak",
            "result": {
                "peak wavelength (nm)": float(wavelength[pk]),
                "peak absorbance":      float(abs_work[pk]),
            },
        }

    # 2+ peaks → bipyramid-style analysis on the top 2
    top2             = peaks[np.argsort(props["prominences"])[-2:]]
    tspr_idx, lspr_idx = sorted(top2, key=lambda i: wavelength[i])

    tspr_wl  = float(wavelength[tspr_idx])
    tspr_abs = float(abs_work[tspr_idx])
    lspr_wl  = float(wavelength[lspr_idx])
    lspr_abs = float(abs_work[lspr_idx])
    abs400   = float(np.interp(400, wavelength, abs_work))

    try:
        fwhm, fwhm_l, fwhm_r = calculate_fwhm(wavelength, abs_work, lspr_idx)
    except ValueError:
        fwhm = fwhm_l = fwhm_r = float("nan")

    status = "2 peaks" if n == 2 else f"{n} peaks (top 2 used)"

    return {
        "n_peaks": n,
        "status":  status,
        "result": {
            "LSPR wavelength (nm)": lspr_wl,
            "LSPR absorbance":      lspr_abs,
            "TSPR wavelength (nm)": tspr_wl,
            "TSPR absorbance":      tspr_abs,
            "LSPR - TSPR":          lspr_abs - tspr_abs,
            "LSPR / TSPR":          lspr_abs / tspr_abs,
            "Absorbance at 400 nm": abs400,
            "TSPR / Abs(400 nm)":   tspr_abs / abs400,
            "LSPR FWHM (nm)":       fwhm,
            "LSPR FWHM left (nm)":  fwhm_l,
            "LSPR FWHM right (nm)": fwhm_r,
        },
    }


def analyze_all(wavelength, spectra, labels, smooth=False):
    """Run analyze_one on every column of spectra, return list of result dicts."""
    results = []
    for label, col in zip(labels, spectra.T):
        entry         = analyze_one(wavelength, col, smooth=smooth)
        entry["label"] = label
        results.append(entry)
    return results


# ---------------------------------------------------------
# Multi-spectrum plotting
# ---------------------------------------------------------

def save_multi_plot(wavelength, spectra, labels, results, input_file):
    """
    Creates one PNG containing every spectrum.

    Blue curve  : spectrum
    Red dot     : detected LSPR peak
    Red line    : FWHM
    """

    n = len(labels)

    ncols = 7
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(13, 5 * nrows), dpi=300)

    for idx, (label, absorbance, analysis) in enumerate(
        zip(labels, spectra.T, results),
        start=1
    ):

        ax = plt.subplot(nrows, ncols, idx)

        # plot spectrum
        ax.plot(wavelength, absorbance, color="royalblue")

        if analysis["n_peaks"] >= 2:

            res = analysis["result"]

            peak_x = res["LSPR wavelength (nm)"]
            peak_y = res["LSPR absorbance"]

            left = res["LSPR FWHM left (nm)"]
            right = res["LSPR FWHM right (nm)"]

            if not np.isnan(left) and not np.isnan(right):

                half = peak_y / 2

                # FWHM line
                ax.plot(
                    [left, right],
                    [half, half],
                    color="red",
                    linewidth=2,
                )

                # Peak marker
                ax.plot(
                    peak_x,
                    peak_y,
                    "ro",
                    markersize=3,
                )

                title = (
                    f"{label}\n"
                    f"FWHM: {res['LSPR FWHM (nm)']:.1f} nm"
                )

            else:
                title = f"{label}\nFWHM: n/a"

        elif analysis["n_peaks"] == 1:

            res = analysis["result"]

            ax.plot(
                res["peak wavelength (nm)"],
                res["peak absorbance"],
                "ro",
                markersize=3,
            )

            title = f"{label}\n1 peak"

        else:
            title = f"{label}\nNo peaks"

        ax.set_title(title, fontsize=8)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.tick_params(labelsize=6)

    fig.tight_layout()

    output_png = os.path.splitext(input_file)[0] + "_FWHM.png"

    print("Saving PNG to:")
    print(os.path.abspath(output_png))

    fig.savefig(output_png, dpi=300)

    plt.close(fig)

    return output_png


# ---------------------------------------------------------
# Single-spectrum report
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
# Multi-spectrum report
# ---------------------------------------------------------

def write_multi_report(results, input_file, smooth=False):
    output = os.path.splitext(input_file)[0] + "_analyzed.txt"

    n_total    = len(results)
    n_no_peaks = sum(1 for r in results if r["n_peaks"] == 0)
    n_one_peak = sum(1 for r in results if r["n_peaks"] == 1)
    n_two_plus = sum(1 for r in results if r["n_peaks"] >= 2)

    with open(output, "w") as f:

        # --- header ---
        f.write("Multi-Spectrum Analysis Results\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Input file    : {input_file}\n")
        f.write(f"Total spectra : {n_total}\n")
        f.write(f"Smoothing     : {'yes' if smooth else 'no'}\n\n")
        f.write(f"No peaks      : {n_no_peaks}\n")
        f.write(f"One peak      : {n_one_peak}\n")
        f.write(f"Two+ peaks    : {n_two_plus}\n")

        # --- summary table ---
        f.write("\n\nSUMMARY\n")
        f.write("-" * 45 + "\n")

        for r in results:
            lbl    = f"Spectrum {r['label']:>4}"
            status = r["status"]
            if r["n_peaks"] == 0:
                f.write(f"{lbl}: {status}\n")
            elif r["n_peaks"] == 1:
                res = r["result"]
                f.write(
                    f"{lbl}: {status:<12}"
                    f"| {res['peak wavelength (nm)']:.1f} nm,"
                    f" abs {res['peak absorbance']:.4f}\n"
                )
            else:
                res = r["result"]
                f.write(
                    f"{lbl}: {status:<12}"
                    f"| TSPR {res['TSPR wavelength (nm)']:.1f} nm,"
                    f" LSPR {res['LSPR wavelength (nm)']:.1f} nm\n"
                )

        # --- detailed section (spectra with at least one peak) ---
        with_peaks = [r for r in results if r["n_peaks"] >= 1]
        if with_peaks:
            f.write("\n\nDETAILED RESULTS\n")
            f.write("-" * 45 + "\n")
            for r in with_peaks:
                f.write(f"\nSpectrum {r['label']}  [{r['status']}]\n")
                for key, value in r["result"].items():
                    if isinstance(value, float) and np.isnan(value):
                        f.write(f"  {key:<32}: n/a\n")
                    else:
                        f.write(f"  {key:<32}: {value:.4f}\n")

    return output


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="UV-Vis spectrum analyzer."
    )
    parser.add_argument("filepath", nargs="?", default=None,
                        help="Spectrum file (.txt single / .csv multi-spectrum)")
    parser.add_argument("--shape",  default="bipyramid")
    parser.add_argument("--smooth", action="store_true",
                        help="Smooth before peak detection")
    args = parser.parse_args()

    filepath = args.filepath
    if filepath is None:
        filepath = input("Spectrum file: ").strip().strip('"').strip("'")

    if not os.path.isfile(filepath):
        raise SystemExit(f"File not found: {filepath}")

    try:
        ext = os.path.splitext(filepath)[1].lower()

        # ---- multi-spectrum CSV ----
        if ext == ".csv":
            print(f"\nLoading multi-spectrum CSV …")
            wavelength, spectra, labels = load_csv_multi(filepath)
            results = analyze_all(wavelength, spectra, labels, smooth=args.smooth)

            n       = len(results)
            n0      = sum(1 for r in results if r["n_peaks"] == 0)
            n1      = sum(1 for r in results if r["n_peaks"] == 1)
            n2      = sum(1 for r in results if r["n_peaks"] >= 2)

            print(f"  {n} spectra | wavelength "
                  f"{wavelength[0]:.0f}–{wavelength[-1]:.0f} nm")
            print(f"  No peaks: {n0} | One peak: {n1} | Two+ peaks: {n2}\n")

            for r in results:
                lbl    = f"Spectrum {r['label']:>4}"
                status = r["status"]
                if r["n_peaks"] == 0:
                    print(f"  {lbl}: {status}")
                elif r["n_peaks"] == 1:
                    res = r["result"]
                    print(f"  {lbl}: {status:<12}"
                          f"| {res['peak wavelength (nm)']:.1f} nm,"
                          f" abs {res['peak absorbance']:.4f}")
                else:
                    res = r["result"]
                    print(f"  {lbl}: {status:<12}"
                          f"| TSPR {res['TSPR wavelength (nm)']:.1f} nm,"
                          f" LSPR {res['LSPR wavelength (nm)']:.1f} nm")

            output = write_multi_report(results, filepath, smooth=args.smooth)
            print(f"\nSaved to: {output}")

            plot_file = save_multi_plot(
                wavelength,
                spectra,
                labels,
                results,
                filepath,
            )

            print(f"Plot saved to: {plot_file}")
            
            
        # ---- single-spectrum txt ----
        else:
            result = analyze_bipyramid(
                *load_spectrum(filepath), smooth=args.smooth
            )
            print("\nResults")
            print("-" * 30)
            for k, v in result.items():
                print(f"{k:<28}: {v:.4f}")
            output = write_report(result, filepath)
            print(f"\nSaved to: {output}")

    except Exception as e:
        print(f"\nAnalysis failed:\n  {e}")

    finally:
        input("\nPress Enter to close...")
