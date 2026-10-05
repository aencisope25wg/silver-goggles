"""
FWHM analyzer for kinetics UV-Vis CSV files.

This is the same code as in the original notebook (baseline_als_optimized,
baseline_correction, estimate_fwhm, plotting loop) — just wrapped so it can
be run from the command line on one or more CSV files.

Each input CSV must have the same layout as the notebook expects:
    index column = wavelength
    every other column = one spectrum (e.g. one timepoint), header = its label

Usage:
    python fwhm_analyzer.py sample1.csv sample2.csv sample3.csv
    python fwhm_analyzer.py *.csv
    python fwhm_analyzer.py sample1.csv --outdir results/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------
# functions from the notebook
# ---------------------------------------------------------

def norm_df(df):
    normed_df = {}
    for i in df:
        column = df.loc[:, i]
        normed = []
        for val in column:
            norm = (val - min(column)) / (max(column) - min(column))
            normed.append(norm)
            normed_df[i] = normed
    new_df = pd.DataFrame(normed_df)
    new_df.columns = df.columns
    new_df.index = df.index
    return new_df


def baseline_als_optimized(y, lam, p, niter=50):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_correction(df, x, y):
    substracted = []
    baselines = []
    for column in df:
        col = df.loc[:, column]
        col_base = baseline_als_optimized(col, x, y)
        substract = col - col_base
        substracted.append(substract)
        baselines.append(col_base)
    corrected = pd.concat(substracted, axis=1)
    baseline = pd.DataFrame(baselines)
    baseline = baseline.transpose()
    baseline.index = df.index
    baseline.columns = df.columns
    return corrected, baseline


def estimate_fwhm(wavelength, intensity):
    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm = wavelength[indices[-1]] - wavelength[indices[0]]
    return fwhm, half_max, wavelength[indices[-1]], wavelength[indices[0]]


# ---------------------------------------------------------
# per-file processing 
# ---------------------------------------------------------

def process_file(filepath, start_lambda=460, end_lambda=850,
                  lam=2000, p=0.00003, outdir=None):

    sample = os.path.splitext(os.path.basename(filepath))[0]
    dest_dir = outdir if outdir else os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nProcessing {filepath} ...")

    df = pd.read_csv(filepath, index_col=0)
    df.columns = df.columns.astype(float)

    # baseline correction for the whole kinetic series
    df_baselinecorr, baselines = baseline_correction(df, lam, p)

    length = len(df_baselinecorr.columns)
    ncols = 7
    nrows = length // ncols + 1

    fig = plt.figure(figsize=(13, 5 * nrows), dpi=300)
    counter = 1
    FWHM_all = []

    for i in df_baselinecorr.columns:

        # calculate FWHM
        wavelength = list(df_baselinecorr.loc[start_lambda:end_lambda, i].index)
        intensity = list(df_baselinecorr.loc[start_lambda:end_lambda, i])
        fwhm, hm, right, left = estimate_fwhm(wavelength, intensity)
        FWHM_all.append(fwhm)

        # create subplots
        ax = plt.subplot(nrows, ncols, counter)

        # plot individual spectra
        ax.plot(df.loc[start_lambda:end_lambda, i], color='firebrick')
        ax.plot(df_baselinecorr.loc[start_lambda:end_lambda, i], color='royalblue')
        ax.plot(baselines.loc[start_lambda:end_lambda, i], color='grey')
        ax.plot([left, right], [hm, hm], color='grey')

        # make title
        ax.set_title(f'{i}min, FWHM: {fwhm}nm', fontsize='x-small')

        counter += 1
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.tight_layout()

    # save list of FWHM to file
    fwhm_csv = os.path.join(dest_dir, f'{sample}_fwhm.csv')
    FWHM = pd.DataFrame({'Time (min)': df.columns, 'FWHM': FWHM_all})
    FWHM.to_csv(fwhm_csv)

    plot_png = os.path.join(dest_dir, f'{sample}_FWHM_all.png')
    fig.savefig(plot_png)
    plt.close(fig)

    print(f"  FWHM values saved to: {fwhm_csv}")
    print(f"  Plot saved to:        {plot_png}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch FWHM analyzer (from notebook logic).")
    parser.add_argument("files", nargs="+", help="One or more kinetics CSV files")
    parser.add_argument("--start", type=float, default=460, help="Start wavelength (default 460)")
    parser.add_argument("--end", type=float, default=850, help="End wavelength (default 850)")
    parser.add_argument("--lam", type=float, default=2000, help="ALS lambda parameter (default 2000)")
    parser.add_argument("--p", type=float, default=0.00003, help="ALS p parameter (default 0.00003)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same folder as each input file)")
    args = parser.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            print(f"Skipping (not found): {f}")
            continue
        process_file(f, start_lambda=args.start, end_lambda=args.end,
                     lam=args.lam, p=args.p, outdir=args.outdir)
