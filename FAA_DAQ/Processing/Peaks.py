#!/usr/bin/env python3
"""
extract_peaks.py — Per-cycle upper/lower peak extraction for cyclic-loading
DAQ files (16 Hz sampling, ~1 Hz loading).

Opens a file picker (multi-select, or takes one or more files as
command-line arguments), and for EACH file independently: segments the
loading cycles, and for EVERY channel finds the upper and lower peak of
EVERY cycle. Because the strain-gauge and voltage/DCDT tasks are not exactly
in phase, each channel gets its own phase-aligned window per cycle: the
channel's lag relative to the reference channel is measured once by circular
cross-correlation, and every cycle window is shifted by that lag before the
max/min are picked. Channels with inverted polarity or half-cycle offsets are
handled the same way.

OUTPUT — for EACH input file, three tab-separated .txt files next to it (or
--outdir), named from that file's own stem:
    <name>_upper_peaks.txt   time_s | cycle | one column per channel (upper)
    <name>_lower_peaks.txt   time_s | cycle | one column per channel (lower)
    <name>_peak_diff.txt     time_s | cycle | one column per channel (upper-lower)
time_s is the start time of the cycle on the reference channel; cycle is the
cycle number (1, 2, 3, ...). Use --with-times to add a <channel>_t column
after each value column in the upper/lower files, giving the exact time_s at
which that channel's peak was picked.

When multiple files are selected/given, each one is processed on its own —
its own cycle segmentation, its own phase lags, its own three output files —
nothing is merged or shared across files.

REQUIREMENTS: numpy, pandas (tkinter ships with Python). No scipy, no
matplotlib needed.

USAGE
    python extract_peaks.py                                  -> file picker (multi-select)
    python extract_peaks.py input.txt                         -> run on one file
    python extract_peaks.py input1.txt input2.txt input3.txt  -> run on each, independently
    python extract_peaks.py input.txt --with-times --outdir C:\\results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def win_long_path(path):
    """Prefix with \\\\?\\ so Windows accepts paths over the 260-char MAX_PATH limit."""
    if sys.platform == "win32":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 15.88          # Hz
CYCLE_HZ     = 1.0         # nominal loading frequency, Hz
SPC          = int(round(SAMPLE_RATE / CYCLE_HZ))   # samples per cycle = 16
RAMP_SECONDS = 0.0         # skip this much at the start (pre-load / ramp)
MAX_LAG      = SPC // 2    # max per-channel phase offset searched (+/- rows)
REF_PREFER   = "DCDT_Beam_B2_Top"   # preferred reference channel
REF_PREFIX   = "DCDT_"              # fallback: mean of these channels


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE SEGMENTATION  (upward midpoint crossings on the reference signal)
# ══════════════════════════════════════════════════════════════════════════════
def find_upcrossings(sig, level):
    """Fractional row indices where sig crosses 'level' going upward."""
    s0, s1 = sig[:-1], sig[1:]
    idx = np.where((s0 < level) & (s1 >= level))[0]
    frac = (level - sig[idx]) / (sig[idx + 1] - sig[idx])
    return idx + frac


def midpoint_level(sig):
    """Robust oscillation midpoint: mean of the 5th/95th percentiles."""
    return (float(np.percentile(sig, 5)) + float(np.percentile(sig, 95))) / 2.0


def segment_cycles(sig, ramp_rows):
    """Cycle-start rows (fractional) from upward midpoint crossings, with
    obviously-too-short gaps (< SPC/2, from noise near the level) merged."""
    body = sig[ramp_rows:]
    level = midpoint_level(body)
    x = find_upcrossings(body, level) + ramp_rows
    if len(x) < 2:
        return x
    keep = [x[0]]
    for xi in x[1:]:
        if xi - keep[-1] >= SPC / 2:      # reject noise re-crossings
            keep.append(xi)
    return np.array(keep)


# ══════════════════════════════════════════════════════════════════════════════
# PER-CHANNEL PHASE LAG  (one circular cross-correlation per channel)
# ══════════════════════════════════════════════════════════════════════════════
def channel_lag(ref, sig, ramp_rows):
    """Integer lag (rows) that best aligns sig to ref over the cyclic region,
    searched in -MAX_LAG..+MAX_LAG. Positive lag = sig events occur LATER."""
    a = ref[ramp_rows:]
    b = sig[ramp_rows:]
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-30 or b.std() < 1e-30:
        return 0
    best_lag, best_r = 0, -np.inf
    n = len(a)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            r = np.dot(a[:n - lag], b[lag:])
        else:
            r = np.dot(a[-lag:], b[:n + lag])
        r = abs(r)                        # inverted channels align too
        if r > best_r:
            best_r, best_lag = r, lag
    return best_lag


# ══════════════════════════════════════════════════════════════════════════════
# PEAK EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_peaks(df, cols, starts):
    """For each cycle window (per-channel phase-shifted), pick upper and
    lower peak value and their row index for every channel.
    Returns dict: upper[c], lower[c], t_up[c], t_lo[c]  (arrays, len n_cycles)."""
    n = len(df)
    ramp_rows = int(RAMP_SECONDS * SAMPLE_RATE)
    ref = build_reference(df, cols)
    lags = {c: channel_lag(ref, df[c].to_numpy(float), ramp_rows) for c in cols}

    n_cyc = len(starts) - 1
    upper = {c: np.full(n_cyc, np.nan) for c in cols}
    lower = {c: np.full(n_cyc, np.nan) for c in cols}
    t_up = {c: np.full(n_cyc, np.nan) for c in cols}
    t_lo = {c: np.full(n_cyc, np.nan) for c in cols}
    tvec = df["time_s"].to_numpy(float)

    for c in cols:
        x = df[c].to_numpy(float)
        L = lags[c]
        for i in range(n_cyc):
            a = int(np.ceil(starts[i])) + L
            b = int(np.ceil(starts[i + 1])) + L
            a = max(a, 0)
            b = min(b, n)
            if b - a < 3:
                continue
            w = x[a:b]
            iu = int(np.argmax(w))
            il = int(np.argmin(w))
            upper[c][i] = w[iu]
            lower[c][i] = w[il]
            t_up[c][i] = tvec[a + iu]
            t_lo[c][i] = tvec[a + il]
    return upper, lower, t_up, t_lo, lags


def build_reference(df, cols):
    if REF_PREFER in df.columns:
        return df[REF_PREFER].to_numpy(float)
    ref_cols = [c for c in cols if c.startswith(REF_PREFIX)]
    if ref_cols:
        return df[ref_cols].to_numpy(float).mean(axis=1)
    return df[cols[0]].to_numpy(float)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def process_file(in_path, outdir, with_times):
    """Run the full peak-extraction pipeline on a single input file and
    write its three output files. Independent of any other file processed
    in the same run — its own cycle segmentation, its own phase lags."""
    df = pd.read_csv(in_path, sep="\t")
    if "time_s" not in df.columns:
        sys.exit("input has no time_s column")
    cols = [c for c in df.columns
            if c != "time_s" and not c.startswith(("inserted_", "repaired_"))]

    print(f"\nInput    : {os.path.basename(in_path)}")
    print(f"Rows     : {len(df):,}  ({df.time_s.iloc[-1]:.2f} s)")
    print(f"Channels : {len(cols)}")

    ramp_rows = int(RAMP_SECONDS * SAMPLE_RATE)
    ref = build_reference(df, cols)
    starts = segment_cycles(ref, ramp_rows)
    if len(starts) < 2:
        sys.exit("no loading cycles found — check RAMP_SECONDS / reference "
                 "channel at the top of the script")
    n_cyc = len(starts) - 1
    gaps = np.diff(starts)
    print(f"Cycles   : {n_cyc}  (gap {gaps.min():.2f}-{gaps.max():.2f} rows, "
          f"nominal {SPC})")

    upper, lower, t_up, t_lo, lags = extract_peaks(df, cols, starts)
    big_lags = {c: l for c, l in lags.items() if abs(l) >= 2}
    if big_lags:
        print("Per-channel phase lags vs reference (rows, |lag|>=2 shown):")
        for c, l in big_lags.items():
            print(f"  {c:30s} {l:+d}")

    cycle_no = np.arange(1, n_cyc + 1)
    cycle_t = starts[:-1] / SAMPLE_RATE

    def make_table(vals, times, with_times):
        d = {"time_s": np.round(cycle_t, 6), "cycle": cycle_no}
        for c in cols:
            d[c] = vals[c]
            if with_times:
                d[c + "_t"] = times[c]
        return pd.DataFrame(d)

    up_df = make_table(upper, t_up, with_times)
    lo_df = make_table(lower, t_lo, with_times)
    diff = {c: upper[c] - lower[c] for c in cols}
    df_df = make_table(diff, None, False)

    file_outdir = outdir if outdir is not None else os.path.dirname(os.path.abspath(in_path))
    os.makedirs(file_outdir, exist_ok=True)
    outdir = file_outdir
    stem = os.path.splitext(os.path.basename(in_path))[0]
    paths = {}
    for tag, tab in [("upper_peaks", up_df), ("lower_peaks", lo_df),
                     ("peak_diff", df_df)]:
        p = os.path.join(outdir, f"{stem}_{tag}.txt")
        tab.to_csv(win_long_path(p), sep="\t", index=False, float_format="%.6f",
                   lineterminator="\r\n")
        paths[tag] = p
        print(f"written: {p}  ({len(tab)} cycles x {len(tab.columns)} cols)")

    # quick sanity summary on the reference channel
    rc = REF_PREFER if REF_PREFER in cols else cols[0]
    print(f"\nSanity ({rc}): mean upper {np.nanmean(upper[rc]):.6g}, "
          f"mean lower {np.nanmean(lower[rc]):.6g}, "
          f"mean range {np.nanmean(diff[rc]):.6g}")
    print("Done.")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[],
                    help="one or more input files; if omitted a file browser opens (multi-select)")
    ap.add_argument("--outdir", default=None,
                    help="output folder for ALL files (default: each file's own folder)")
    ap.add_argument("--with-times", action="store_true",
                    help="add a <channel>_t column after each value column in "
                         "the upper/lower files (exact pick time per channel)")
    args = ap.parse_args()

    last_dir_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  ".last_dir.txt")

    in_paths = list(args.inputs)
    if not in_paths:
        import tkinter as tk
        from tkinter import filedialog
        start_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isfile(last_dir_file):
            saved = open(last_dir_file).read().strip()
            if os.path.isdir(saved):
                start_dir = saved
        root = tk.Tk()
        root.withdraw()
        in_paths = list(filedialog.askopenfilenames(
            title="Select cyclic-loading data file(s)",
            initialdir=start_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]))
        root.destroy()
        if not in_paths:
            sys.exit("No file selected.")
        with open(last_dir_file, "w") as fh:
            fh.write(os.path.dirname(os.path.abspath(in_paths[-1])))

    print(f"Selected {len(in_paths)} file(s); each is processed independently:")
    for p in in_paths:
        print(f"  {os.path.basename(p)}")

    for in_path in in_paths:
        process_file(in_path, args.outdir, args.with_times)


if __name__ == "__main__":
    main()