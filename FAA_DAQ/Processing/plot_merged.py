"""
plot_merged.py — Multi-file loader + Strain / DCDT / Pressure time-series viewer.

Select multiple DAQ files (raw, processed, trimmed, ... any tab-separated
file with a time_s column). Files are sorted by filename (natural numeric
order, e.g. Set_2 before Set_10) and concatenated end-to-end in that order —
placed one after another, no trimming, no continuity matching. A continuous
time_s column is regenerated for the merged data at SAMPLE_RATE.

Column groups (from the FIRST selected file — all files are assumed to
share the same columns):
  • Strain gauges     (SG_*)
  • DCDT displacement (DCDT_*)
  • Pressure/voltage  ("pressure" in name, or volt_ch — excluding DCDT_)

Plots the three groups as separate panels stacked on a shared time axis.
A button toggles that shared axis between linear and log scale.

Usage:
    python plot_merged.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox

SAMPLE_RATE = 16  # Hz — used to regenerate a continuous time_s column after merging

# 24-colour palette (same as the other Processing/ scripts)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]


def is_dcdt_col(c):
    return c.startswith("DCDT_")


def is_strain_col(c):
    return c.startswith("SG_")


def is_pressure_col(c):
    return not is_dcdt_col(c) and ("pressure" in c.lower() or "volt_ch" in c.lower())


def natural_key(path):
    """Sort key that orders embedded numbers numerically (Set_2 before Set_10)."""
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def main():
    root = tk.Tk()
    root.withdraw()

    in_paths = filedialog.askopenfilenames(
        title="Select files to merge and plot (any filename)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not in_paths:
        messagebox.showinfo("Cancelled", "No files selected. Exiting.")
        root.destroy()
        return
    root.destroy()

    in_paths = sorted(in_paths, key=natural_key)
    print(f"\nSelected {len(in_paths)} file(s), placed one after another in this order:")
    for p in in_paths:
        print(f"  {os.path.basename(p)}")

    strain_cols = dcdt_cols = press_cols = data_cols = None
    chunks = []
    file_boundaries = []   # (filename, cumulative row count at end of this file)
    row_count = 0
    for p in in_paths:
        df = pd.read_csv(p, sep="\t")
        if data_cols is None:
            all_cols    = [c for c in df.columns if c != "time_s"]
            strain_cols = [c for c in all_cols if is_strain_col(c)]
            dcdt_cols   = [c for c in all_cols if is_dcdt_col(c)]
            press_cols  = [c for c in all_cols if is_pressure_col(c)]
            data_cols   = strain_cols + dcdt_cols + press_cols
            missing = [c for c in all_cols if c not in data_cols]
            if missing:
                print(f"  Note: unclassified columns ignored: {missing}")
        chunks.append(df[data_cols].reset_index(drop=True))
        row_count += len(df)
        file_boundaries.append((os.path.basename(p), row_count))
        print(f"  {os.path.basename(p)}: {len(df):,} rows")

    merged = pd.concat(chunks, ignore_index=True)
    merged.insert(0, "time_s", np.arange(len(merged)) / SAMPLE_RATE)
    print(f"\nMerged: {len(merged):,} rows total ({merged['time_s'].iloc[-1]:.2f} s)")

    t = merged["time_s"].to_numpy()
    # last file's boundary is the end of the data, not a separator worth drawing
    boundary_times = [(name, t[row - 1]) for name, row in file_boundaries[:-1]]
    if boundary_times:
        print(f"\nFile separators ({len(boundary_times)}):")
        for name, tb in boundary_times:
            print(f"  {tb:.4f} s  — end of {name}")
    else:
        print("\nOnly one file (or none) — no separator lines to draw.")

    groups = [
        ("Strain (SG_*)",      strain_cols),
        ("DCDT displacement",  dcdt_cols),
        ("Pressure / Voltage", press_cols),
    ]
    groups = [(title, cols) for title, cols in groups if cols]
    if not groups:
        messagebox.showerror("No data", "None of the selected files have recognizable "
                                          "SG_/DCDT_/pressure columns.")
        return

    fig, axes = plt.subplots(len(groups), 1, figsize=(15, 9), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("Merged data — Strain / DCDT / Pressure vs time",
                 fontsize=11, fontweight="bold")

    for ax, (title, cols) in zip(axes, groups):
        for i, c in enumerate(cols):
            ax.plot(t, merged[c].to_numpy(float), lw=1.0,
                    color=_PALETTE[i % len(_PALETTE)], label=c, alpha=0.85)
        for name, tb in boundary_times:
            ax.axvline(tb, color="red", lw=1.5, ls="-", alpha=0.9, zorder=20)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=min(len(cols), 6), loc="upper right")

    # file-separator labels, drawn once along the top panel only
    for name, tb in boundary_times:
        axes[0].annotate(name, xy=(tb, 1.0), xycoords=("data", "axes fraction"),
                          xytext=(3, -3), textcoords="offset points",
                          fontsize=6, color="black", rotation=90,
                          ha="left", va="top")

    axes[-1].set_xlabel("time_s")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.15)

    # ── Linear / Log time-axis toggle ──────────────────────────────────────
    btn_ax = fig.add_axes([0.01, 0.94, 0.11, 0.04])
    btn = Button(btn_ax, "Log time axis", color="#e8e8e8", hovercolor="#d0d0d0")
    btn.label.set_fontsize(8)
    state = {"log": False}

    def toggle_scale(_):
        state["log"] = not state["log"]
        scale = "log" if state["log"] else "linear"
        for ax in axes:
            ax.set_xscale(scale)
        btn.label.set_text("Linear time axis" if state["log"] else "Log time axis")
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_scale)

    plt.show()


if __name__ == "__main__":
    main()
