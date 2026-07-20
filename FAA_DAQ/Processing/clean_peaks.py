#!/usr/bin/env python3
"""
clean_peaks.py — Remove big jumps and outlier cycles from Peaks.py's output
files (<name>_upper_peaks.txt, _lower_peaks.txt, _peak_diff.txt), per channel.

Two kinds of defects are handled independently, per channel:
  1. OUTLIER / SPIKE  — a single (or few) cycle whose peak value is way off
     from its neighbors and then returns to the previous level. Detected with
     a rolling Hampel filter (local median + MAD) and fixed by linear
     interpolation from the nearest good cycles.
  2. JUMP / STEP       — a persistent baseline shift that does NOT revert
     (e.g. left over from a merge-file boundary). Detected as an abrupt
     single-cycle change in the series that is confirmed by the following
     cycles staying at the new level, and fixed by subtracting the shift from
     every cycle after it — this preserves the real shape/trend of the data,
     it just removes the artificial step.

Genuine gradual trends (e.g. progressive residual displacement growing over
thousands of cycles) are NOT touched — both detectors only fire on abrupt,
single-cycle-scale changes relative to the channel's own local noise level.

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
JUMP_K        = 6.0   # flag a cycle-to-cycle step this many MADs above typical step size
JUMP_CONFIRM  = 5      # cycles after a candidate jump checked to confirm it doesn't revert
JUMP_CONFIRM_FRAC = 0.5  # the confirmed shift must retain at least this fraction of the jump


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


def fix_jumps(x, k=JUMP_K, confirm=JUMP_CONFIRM, confirm_frac=JUMP_CONFIRM_FRAC):
    """Detect persistent single-cycle level shifts (not reverted by the
    following `confirm` cycles) and remove them by subtracting the shift
    from every value after the jump. Returns (cleaned_x, jump_events) where
    jump_events is a list of (cycle_index, shift_amount)."""
    x = x.copy()
    n = len(x)
    jumps = []
    # a single global noise estimate over the whole (already despiked) series —
    # a small local window can land on a quiet stretch and underestimate
    # noise, causing false positives on ordinary trend/slope changes
    mad_d = robust_mad(np.diff(x))
    if mad_d < 1e-12:
        return x, jumps
    i = 0
    while i < n - 1:
        step = x[i + 1] - x[i]
        if abs(step) > k * mad_d:
            after = x[i + 1: i + 1 + confirm]
            after = after[~np.isnan(after)]
            if len(after) >= 2 and abs(np.median(after) - x[i]) > confirm_frac * abs(step):
                shift = np.median(after) - x[i]
                x[i + 1:] -= shift
                jumps.append((i + 1, shift))
                mad_d = robust_mad(np.diff(x))  # refresh after correcting
        i += 1
    return x, jumps


def clean_channel(x):
    x = np.asarray(x, dtype=float)
    x, spikes = despike(x)
    x, jumps = fix_jumps(x)
    return x, spikes, jumps


def clean_file(path, out_dir):
    df = pd.read_csv(path, sep="\t")
    skip_cols = {"time_s", "cycle"}
    channel_cols = [c for c in df.columns
                    if c not in skip_cols and not c.endswith("_t")]

    print(f"\n{os.path.basename(path)}: {len(df)} cycles, {len(channel_cols)} channel(s)")
    total_spikes = total_jumps = 0
    for c in channel_cols:
        cleaned, spikes, jumps = clean_channel(df[c].to_numpy())
        df[c] = cleaned
        if spikes:
            print(f"  {c:30s} {len(spikes)} spike(s) at cycles "
                  f"{[df['cycle'].iloc[i] for i in spikes]}")
        if jumps:
            detail = ", ".join(f"cycle {df['cycle'].iloc[i]}: {shift:+.5f}"
                                for i, shift in jumps)
            print(f"  {c:30s} {len(jumps)} jump(s) removed — {detail}")
        total_spikes += len(spikes)
        total_jumps += len(jumps)

    if total_spikes == 0 and total_jumps == 0:
        print("  No spikes or jumps detected.")

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
