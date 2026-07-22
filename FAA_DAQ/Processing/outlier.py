#!/usr/bin/env python3
"""
clean_peaks.py — Remove outlier cycles from Peaks.py's output files
(<name>_upper_peaks.txt, _lower_peaks.txt, _peak_diff.txt), per channel.

OUTLIER / SPIKE  — a single (or few) cycle whose peak value is way off from
its neighbors and then returns to the previous level. Detected with a
rolling Hampel filter (local median + MAD) and fixed by linear interpolation
from the nearest good cycles.

Genuine gradual trends (e.g. progressive residual displacement growing over
thousands of cycles) and persistent baseline shifts/jumps are NOT touched —
the detector only fires on abrupt, single-cycle-scale changes that revert,
relative to the channel's own local noise level.

"cycle" and "time_s" (and any "<channel>_t" companion columns from
--with-times) are carried through unchanged.

Usage:
    python clean_peaks.py                 -> file picker (multi-select)
    python clean_peaks.py file1.txt file2.txt
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
SPIKE_WINDOW  = 11    # cycles, rolling window (odd) for the Hampel spike filter
SPIKE_K       = 6.0   # flag a cycle as a spike if it's this many MADs from the local median


def robust_mad(a):
    """Median absolute deviation, scaled to be comparable to a std-dev."""
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return 0.0
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med))


def despike(x, window=SPIKE_WINDOW, k=SPIKE_K):
    """Hampel filter: replace isolated outlier cycles with NaN (to be
    interpolated by the caller). Returns (cleaned_x, spike_indices)."""
    x = x.copy()
    n = len(x)
    half = window // 2
    spikes = []
    for i in range(n):
        if np.isnan(x[i]):
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighborhood = np.delete(x[lo:hi], i - lo)
        neighborhood = neighborhood[~np.isnan(neighborhood)]
        if len(neighborhood) < 3:
            continue
        med = np.median(neighborhood)
        mad = 1.4826 * np.median(np.abs(neighborhood - med))
        if mad < 1e-12:
            continue
        if abs(x[i] - med) > k * mad:
            spikes.append(i)
            x[i] = np.nan
    if spikes:
        idx = np.arange(n)
        good = ~np.isnan(x)
        if good.sum() >= 2:
            x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x, spikes


def clean_channel(x):
    x = np.asarray(x, dtype=float)
    x, spikes = despike(x)
    return x, spikes


def clean_file(path, out_dir):
    df = pd.read_csv(path, sep="\t")
    skip_cols = {"time_s", "cycle"}
    channel_cols = [c for c in df.columns
                    if c not in skip_cols and not c.endswith("_t")]

    print(f"\n{os.path.basename(path)}: {len(df)} cycles, {len(channel_cols)} channel(s)")
    total_spikes = 0
    for c in channel_cols:
        cleaned, spikes = clean_channel(df[c].to_numpy())
        df[c] = cleaned
        if spikes:
            print(f"  {c:30s} {len(spikes)} spike(s) at cycles "
                  f"{[df['cycle'].iloc[i] for i in spikes]}")
        total_spikes += len(spikes)

    if total_spikes == 0:
        print("  No spikes detected.")

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_cleaned.txt")
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"  Written: {out_path}")
    return out_path


def main():
    paths = sys.argv[1:]
    if not paths:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select peak file(s) to clean (upper/lower/diff)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        root.destroy()
        if not paths:
            sys.exit("No file(s) selected.")

    out_dir = os.path.dirname(os.path.abspath(paths[0]))
    for p in paths:
        clean_file(p, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
