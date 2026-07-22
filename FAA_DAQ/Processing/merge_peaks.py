"""
merge_peaks.py — Straight concatenation of Peaks.py output files, no
trimming, no interactive markers, no continuity offset matching.

Select two or more peak files of the SAME kind (all _upper_peaks.txt, or all
_lower_peaks.txt, or all _peak_diff.txt) and they are stacked end-to-end in
the order selected. Only two columns are touched:
  "cycle"   renumbered 1..N across the merge
  "time_s"  kept continuously increasing (each file's own time_s stream is
            shifted to start right where the previous file's left off, using
            that file's own median cycle spacing)
Every channel value is copied through unchanged.

Usage:
    python merge_peaks.py
    python merge_peaks.py file1.txt file2.txt ...
"""

import os
import re
import sys
import numpy as np
import pandas as pd


def natural_key(path):
    """Sort key that orders embedded numbers numerically (Set_2 before Set_10)."""
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def main():
    paths = sys.argv[1:]
    if not paths:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select peak files to merge (same kind: upper, lower, or diff)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        root.destroy()
        if not paths:
            sys.exit("No file(s) selected.")

    paths = sorted(paths, key=natural_key)
    print(f"\nMerging {len(paths)} file(s):")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    chunks = []
    t_offset = 0.0
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        if "time_s" in df.columns:
            df = df.copy()
            dt = np.median(np.diff(df["time_s"].to_numpy(float))) if len(df) > 1 else 1.0
            df["time_s"] = df["time_s"] - df["time_s"].iloc[0] + t_offset
            t_offset = df["time_s"].iloc[-1] + dt
        chunks.append(df)

    merged = pd.concat(chunks, ignore_index=True)
    if "cycle" in merged.columns:
        merged["cycle"] = np.arange(1, len(merged) + 1)

    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    out_path = filedialog.asksaveasfilename(
        title="Save merged peaks file as",
        initialdir=os.path.dirname(os.path.abspath(paths[0])),
        initialfile="merged_peaks.txt",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    root.destroy()
    if not out_path:
        sys.exit("No save location selected.")

    merged.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nDone — {len(merged)} rows written to:\n  {out_path}")


if __name__ == "__main__":
    main()
