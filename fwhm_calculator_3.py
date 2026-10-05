#!/usr/bin/env python3
"""
fwhm_calculator.py

Calculate the Full Width at Half Maximum (FWHM) of extinction/absorbance
bands from a CSV spectrum file, and save annotated grid images (no CSV
output).

Expected CSV layout
--------------------
Column 1        : wavelength (or wavenumber, energy, etc.)
Column 2, 3, ... : one or more extinction/absorbance spectra (one column
                    per sample/curve)

The script automatically detects whether each spectrum has one or two
resolvable bands (peaks) -- labeled LSPR (higher wavelength) and TSPR
(lower wavelength) -- and reports their positions, heights, FWHM (LSPR
only), and derived metrics (LSPR-TSPR, LSPR/TSPR, abs400, LSPR/abs400).
Peak-detection is always restricted to wavelengths between 450 and 1050
nm, since a band's true peak commonly falls outside the measured range
below 450 nm (and would otherwise be mistaken for real data), and some
instruments show a small noise spike near the 1080-1100 nm edge that
could otherwise be mistaken for a real (and wrongly labeled LSPR) band. All results are drawn
onto grouped grid images -- there is no CSV output.

Usage
-----
    python fwhm_calculator.py input.csv
    python fwhm_calculator.py input.csv --prominence 0.02 --max-bands 2
    python fwhm_calculator.py input.csv --max-per-image 20 --grid-cols 5

Dependencies: pandas, numpy, scipy, matplotlib
    pip install pandas numpy scipy matplotlib
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


def detect_bands(x, y, max_bands=2, prominence_frac=0.03, min_distance_frac=0.02, min_height=None):
    """
    Detect up to `max_bands` peaks (bands) in a spectrum.

    Parameters
    ----------
    x, y : 1D arrays (wavelength, intensity)
    max_bands : maximum number of bands to keep (1 or 2 in this use case,
                but the function works for any value)
    prominence_frac : minimum peak prominence, as a fraction of THIS
                       spectrum's own amplitude (max - min). This makes the
                       detector adapt to each spectrum's own contrast, but
                       on its own it can't tell noise from a real small band
                       when a spectrum has barely developed any signal yet.
    min_distance_frac : minimum distance between two peaks, as a fraction
                         of the number of points, to avoid picking noise
                         spikes right next to a real peak as a second band.
    min_height : optional absolute height floor (same units as y). Any
                 candidate peak below this value is rejected regardless of
                 its prominence. Pass this in as a fraction of the *whole
                 dataset's* maximum (not just this spectrum's) so that
                 near-baseline/noisy spectra don't generate false bands
                 just because the sensitivity was lowered to catch a real
                 but small band elsewhere in the dataset.

    Returns
    -------
    peak_idx : indices (into x/y) of the detected peaks, sorted by
               prominence (strongest first) then trimmed to max_bands,
               finally re-sorted left-to-right for readability.
    """
    amplitude = np.nanmax(y) - np.nanmin(y)
    if amplitude <= 0 or np.all(np.isnan(y)):
        return np.array([], dtype=int)

    min_prominence = prominence_frac * amplitude
    min_distance = max(1, int(min_distance_frac * len(y)))

    peak_idx, props = find_peaks(y, prominence=min_prominence, distance=min_distance)

    if len(peak_idx) == 0:
        # Fallback: relax prominence once in case the band is very broad
        # and shallow (common for weak extinction bands).
        peak_idx, props = find_peaks(y, prominence=min_prominence * 0.3, distance=min_distance)

    if len(peak_idx) == 0:
        return np.array([], dtype=int)

    # Reject candidates that don't clear the absolute noise floor, e.g.
    # tiny wiggles in a spectrum that hasn't developed real signal yet.
    if min_height is not None:
        keep = y[peak_idx] >= min_height
        peak_idx = peak_idx[keep]
        props["prominences"] = props["prominences"][keep]
        if len(peak_idx) == 0:
            return np.array([], dtype=int)

    # keep the most prominent peaks, up to max_bands
    order = np.argsort(props["prominences"])[::-1]
    peak_idx = peak_idx[order][:max_bands]

    return np.sort(peak_idx)


def compute_fwhm(x, y, peak_idx):
    """
    Compute FWHM for each peak using scipy's peak_widths at rel_height=0.5.

    peak_widths finds, for each peak, the width of the peak at half its
    prominence (measured from an interpolated local baseline), and returns
    the result in units of samples. We convert to x-axis units using the
    local sample spacing, which correctly handles non-uniform x-axes too
    since we interpolate the crossing points back onto x.

    Returns a list of dicts with peak position, FWHM, and the interpolated
    left/right half-maximum crossing points (useful for plotting).
    """
    results = []
    if len(peak_idx) == 0:
        return results

    widths, width_heights, left_ips, right_ips = peak_widths(y, peak_idx, rel_height=0.5)

    # left_ips / right_ips are fractional indices; interpolate onto the
    # real x-axis (handles non-uniform wavelength spacing correctly).
    idx_all = np.arange(len(x))
    for i, p in enumerate(peak_idx):
        x_left = np.interp(left_ips[i], idx_all, x)
        x_right = np.interp(right_ips[i], idx_all, x)
        fwhm = abs(x_right - x_left)
        results.append({
            "peak_index": int(p),
            "peak_x": x[p],
            "peak_y": y[p],
            "half_max_level": width_heights[i],
            "x_left": x_left,
            "x_right": x_right,
            "fwhm": fwhm,
        })
    return results


def compute_band_ratio(bands):
    """
    Ratio of extinction maxima: higher-wavelength band / lower-wavelength band.

    `bands` is assumed to be ordered from lowest to highest peak wavelength
    (this holds naturally here because detect_bands returns peaks sorted by
    index, and wavelength is monotonically increasing with index).

    Returns None if there aren't exactly two bands to compare (ratio is
    only meaningful when two bands are present).
    """
    if len(bands) != 2:
        return None
    y_low_wl = bands[0]["peak_y"]
    y_high_wl = bands[-1]["peak_y"]
    if y_low_wl == 0:
        return None
    return y_high_wl / y_low_wl


def assign_lspr_tspr(bands):
    """
    Map detected bands onto the LSPR / TSPR naming used for two-band
    plasmonic (e.g. gold nanorod) spectra:
      - LSPR (Longitudinal SPR): the higher-wavelength band
      - TSPR (Transverse SPR):   the lower-wavelength band

    If only one band is found, it's assigned to LSPR (the main, usually
    dominant plasmon band) and TSPR is left as None -- there's no second
    band to call TSPR. If no bands are found, both are None.
    """
    if len(bands) == 0:
        return None, None
    if len(bands) == 1:
        return bands[0], None
    # 2+ bands: bands is sorted ascending by wavelength already
    return bands[-1], bands[0]


def compute_metrics(bands, band_ratio, x_raw, y_raw, abs400_wavelength=400.0):
    """
    Build the full set of requested per-spectrum metrics.

    x_raw, y_raw : the FULL, uncropped wavelength/intensity arrays for this
                   spectrum, used only to read off abs400 -- this way abs400
                   is still correct even if --wl-min cropped 400 nm out of
                   the peak-search window.

    Returns a dict with keys:
        lspr_wavelength, lspr_absorbance,
        tspr_wavelength, tspr_absorbance,
        lspr_minus_tspr, lspr_over_tspr,
        abs400, lspr_over_abs400,
        fwhm_lspr, fwhm_tspr
    Any value that can't be determined (e.g. TSPR when only one band was
    found) is set to None.
    """
    lspr, tspr = assign_lspr_tspr(bands)

    metrics = {
        "lspr_wavelength": lspr["peak_x"] if lspr else None,
        "lspr_absorbance": lspr["peak_y"] if lspr else None,
        "fwhm_lspr": lspr["fwhm"] if lspr else None,
        "tspr_wavelength": tspr["peak_x"] if tspr else None,
        "tspr_absorbance": tspr["peak_y"] if tspr else None,
    }

    if lspr and tspr:
        metrics["lspr_minus_tspr"] = lspr["peak_x"] - tspr["peak_x"]
        metrics["lspr_over_tspr"] = band_ratio  # same quantity, already computed
    else:
        metrics["lspr_minus_tspr"] = None
        metrics["lspr_over_tspr"] = None

    # abs400 is read from the raw, uncropped spectrum so it's correct
    # regardless of any --wl-min/--wl-max cropping used for peak-finding.
    if x_raw.min() <= abs400_wavelength <= x_raw.max():
        abs400 = float(np.interp(abs400_wavelength, x_raw, y_raw))
    else:
        abs400 = None
    metrics["abs400"] = abs400

    if lspr and abs400 not in (None, 0):
        metrics["lspr_over_abs400"] = lspr["peak_y"] / abs400
    else:
        metrics["lspr_over_abs400"] = None

    return metrics


def analyze_spectrum(x, y, label, max_bands, prominence_frac, min_height=None):
    y_clean = np.asarray(y, dtype=float)
    valid = ~np.isnan(y_clean)
    if valid.sum() < 5:
        print(f"[{label}] Not enough valid data points, skipping.")
        return [], None

    x_valid = np.asarray(x, dtype=float)[valid]
    y_valid = y_clean[valid]

    peak_idx = detect_bands(x_valid, y_valid, max_bands=max_bands, prominence_frac=prominence_frac,
                             min_height=min_height)

    if len(peak_idx) == 0:
        print(f"[{label}] No clear band detected.")
        return [], None

    bands = compute_fwhm(x_valid, y_valid, peak_idx)
    band_ratio = compute_band_ratio(bands)

    print(f"[{label}] Detected {len(bands)} band(s):")
    for i, b in enumerate(bands, start=1):
        print(f"    Band {i}: peak at {b['peak_x']:.2f}, "
              f"max extinction = {b['peak_y']:.4f}, "
              f"FWHM = {b['fwhm']:.3f}, "
              f"half-max window = [{b['x_left']:.2f}, {b['x_right']:.2f}]")
    if band_ratio is not None:
        print(f"    Band ratio (higher-wavelength / lower-wavelength) = {band_ratio:.4f}")

    return bands, band_ratio


def _fmt(value, fmt="{:.2f}"):
    """Small helper: format a number, or show '--' if it's None (metric not available)."""
    return fmt.format(value) if value is not None else "--"


def plot_grid(entries, out_dir, cols=4, max_per_image=12):
    """
    Plot many spectra as small subplots grouped into a handful of PNG files,
    instead of one file per spectrum.

    entries : list of tuples (x_full, y_full, label, bands, band_ratio, metrics)
              x_full/y_full are the RAW, uncropped spectrum -- plotting the
              full curve (rather than just the cropped analysis window) lets
              us also mark the abs400 point even when peak-detection itself
              was restricted to a narrower wavelength range.
    cols    : number of subplot columns per image
    max_per_image : how many spectra to pack into a single PNG before
                    starting a new one

    Returns the list of file paths written.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    for start in range(0, len(entries), max_per_image):
        chunk = entries[start:start + max_per_image]
        n = len(chunk)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (x_full, y_full, label, bands, band_ratio, metrics) in zip(axes_flat, chunk):
            ax.plot(x_full, y_full, color="black", lw=1.0)

            # Lock the axis limits to the data range now, before adding any
            # markers/guide-lines, so those additions can't stretch the axes.
            ax.relim()
            ax.autoscale_view()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            def _guide_lines(px, py, color):
                """Dotted guide lines from each axis to the point -- and no further."""
                ax.vlines(px, ylim[0], py, color=color, linestyle=":", linewidth=0.8)
                ax.hlines(py, xlim[0], px, color=color, linestyle=":", linewidth=0.8)

            # LSPR (the last band in the low-to-high-wavelength ordered list):
            # marked with its peak point, guide lines, AND its half-max width
            # line, since that FWHM is the one that's wanted.
            if metrics["lspr_wavelength"] is not None:
                lx, ly = metrics["lspr_wavelength"], metrics["lspr_absorbance"]
                _guide_lines(lx, ly, "#d62728")
                ax.plot(lx, ly, "o", color="#d62728", ms=5, zorder=5)
                lspr_band = bands[-1]
                ax.hlines(lspr_band["half_max_level"], lspr_band["x_left"], lspr_band["x_right"],
                           color="#d62728", linestyle="--", linewidth=1.2)

            # TSPR: peak point + guide lines, no FWHM line needed.
            if metrics["tspr_wavelength"] is not None:
                tx, ty = metrics["tspr_wavelength"], metrics["tspr_absorbance"]
                _guide_lines(tx, ty, "#1f77b4")
                ax.plot(tx, ty, "o", color="#1f77b4", ms=5, zorder=5)

            # abs400: same dot style, with its own guide lines.
            if metrics["abs400"] is not None:
                ax_, ay = 400, metrics["abs400"]
                _guide_lines(ax_, ay, "#2ca02c")
                ax.plot(ax_, ay, "o", color="#2ca02c", ms=5, zorder=5)

            # re-lock limits in case the guide lines nudged autoscale
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=6)

            info_lines = [
                f"LSPR {_fmt(metrics['lspr_wavelength'], '{:.0f}')}nm  A={_fmt(metrics['lspr_absorbance'], '{:.3f}')}  FWHM={_fmt(metrics['fwhm_lspr'], '{:.1f}')}",
                f"TSPR {_fmt(metrics['tspr_wavelength'], '{:.0f}')}nm  A={_fmt(metrics['tspr_absorbance'], '{:.3f}')}",
                f"LSPR-TSPR={_fmt(metrics['lspr_minus_tspr'], '{:.0f}')}  LSPR/TSPR={_fmt(metrics['lspr_over_tspr'], '{:.2f}')}",
                f"abs400={_fmt(metrics['abs400'], '{:.3f}')}  LSPR/abs400={_fmt(metrics['lspr_over_abs400'], '{:.2f}')}",
            ]
            ax.text(0.98, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                    fontsize=6, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))

        # hide any unused subplot slots on the last page
        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.tight_layout()
        page_num = start // max_per_image + 1
        out_path = os.path.join(out_dir, f"fwhm_grid_{page_num:02d}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)

    return written


def parse_time_label(label):
    """
    Extract a numeric x-axis value from a spectrum's column header, so
    trend plots can be drawn as time/step (x) vs. value (y).

    Handles plain numbers ("1", "0.25", "24.1") as well as numbers with
    a trailing unit suffix ("1min", "2.5min"). Returns None if no number
    could be found at the start of the label -- that spectrum is then
    left out of the trend plots rather than guessed at.
    """
    import re
    match = re.match(r'^\s*(-?\d+\.?\d*)', str(label))
    if match:
        return float(match.group(1))
    return None


def plot_trends(rows, out_dir):
    """
    Plot each key metric across the series as a time/step (x) vs value (y)
    chart, all combined into a single grid image (one panel per metric).

    rows : list of dicts, each with a 'time' key plus the metrics from
           compute_metrics(). Rows without a parsable time are skipped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["time"] is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["time"])
    times = [r["time"] for r in rows]

    metric_panels = [
        ("lspr_wavelength", "LSPR wavelength (nm)"),
        ("tspr_wavelength", "TSPR wavelength (nm)"),
        ("lspr_absorbance", "LSPR absorbance"),
        ("tspr_absorbance", "TSPR absorbance"),
        ("fwhm_lspr", "LSPR FWHM (nm)"),
        ("lspr_minus_tspr", "LSPR - TSPR (nm)"),
        ("lspr_over_tspr", "LSPR / TSPR"),
        ("abs400", "abs400"),
        ("lspr_over_abs400", "LSPR / abs400"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes_flat = axes.flatten()
    for ax, (key, title) in zip(axes_flat, metric_panels):
        values = [r.get(key) for r in rows]
        values = [np.nan if v is None else v for v in values]
        ax.plot(times, values, marker="o", ms=3, lw=1, color="#1f77b4")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time / step (from column header)", fontsize=6.5)
        ax.tick_params(labelsize=6.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "trends.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Calculate FWHM of one or two bands in extinction spectra.")
    parser.add_argument("csv_file", help="Path to input CSV file (col 1 = wavelength, others = spectra)")
    parser.add_argument("--max-bands", type=int, default=2, help="Maximum number of bands to detect per spectrum (default: 2)")
    parser.add_argument("--prominence", type=float, default=0.03,
                         help="Minimum peak prominence as a fraction of the spectrum amplitude (default: 0.03)")
    parser.add_argument("--min-height-frac", type=float, default=0.01,
                         help="Reject candidate peaks below this fraction of the WHOLE dataset's maximum value "
                              "(default: 0.01, i.e. 1%%). This is a noise floor: it lets you lower --prominence "
                              "to catch a small real band without picking up noise in spectra that haven't "
                              "developed real signal yet.")
    parser.add_argument("--abs400-wavelength", type=float, default=400.0,
                         help="Wavelength (in the RAW, uncropped data) used for the 'abs400' reference "
                              "absorbance reading (default: 400)")
    parser.add_argument("--grid-cols", type=int, default=4, help="Number of subplot columns per grid image (default: 4)")
    parser.add_argument("--max-per-image", type=int, default=12,
                         help="How many spectra to pack into a single PNG before starting a new one (default: 12)")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs (default: alongside input file)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if df.shape[1] < 2:
        sys.exit("CSV must have at least 2 columns: wavelength + one spectrum.")

    wavelength_col = df.columns[0]
    x_full = df[wavelength_col].to_numpy(dtype=float)
    spectrum_cols = df.columns[1:]

    # Peak-detection is always restricted to 450-1050 nm: below 450, a
    # band's true peak commonly falls outside the measured range and looks
    # like a rising edge; above ~1050, some spectrometers show a small
    # detector-edge noise spike that can otherwise get mistaken for a real
    # (and wrongly labeled LSPR) band.
    WL_MIN = 450.0
    WL_MAX = 1050.0
    mask = (x_full >= WL_MIN) & (x_full <= WL_MAX)
    x = x_full[mask]

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv_file))
    os.makedirs(out_dir, exist_ok=True)

    # Noise floor shared across all spectra: a candidate peak must clear
    # this absolute height, computed once from the whole dataset, so that
    # lowering --prominence to catch a small real band doesn't also let
    # noise through in spectra that haven't developed real signal yet.
    global_max = np.nanmax(df[spectrum_cols].to_numpy(dtype=float)[mask, :])
    min_height = args.min_height_frac * global_max

    plot_entries = []  # collected here, plotted as grouped grids at the end
    trend_rows = []    # collected here, plotted as time/step vs value at the end
    for col in spectrum_cols:
        y_full_col = df[col].to_numpy(dtype=float)  # uncropped, used for abs400
        y = y_full_col[mask]
        bands, band_ratio = analyze_spectrum(x, y, label=str(col), max_bands=args.max_bands,
                                              prominence_frac=args.prominence, min_height=min_height)

        metrics = compute_metrics(bands, band_ratio, x_full, y_full_col,
                                   abs400_wavelength=args.abs400_wavelength)

        trend_row = {"time": parse_time_label(col)}
        trend_row.update(metrics)
        trend_rows.append(trend_row)

        if bands:
            plot_entries.append((x_full, y_full_col, str(col), bands, band_ratio, metrics))

    if plot_entries:
        image_paths = plot_grid(plot_entries, out_dir, cols=args.grid_cols, max_per_image=args.max_per_image)
        print(f"\n{len(image_paths)} grid image(s) saved:")
        for p in image_paths:
            print(f"    {p}")
    else:
        print("\nNo bands detected in any spectrum.")

    trend_path = plot_trends(trend_rows, out_dir)
    if trend_path:
        print(f"\nTrend plot saved to: {trend_path}")
    else:
        print("\nNo trend plot generated (column headers had no parsable numeric time/step value).")


if __name__ == "__main__":
    main()
