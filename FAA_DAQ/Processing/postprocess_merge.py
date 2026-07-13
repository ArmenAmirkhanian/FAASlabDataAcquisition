"""
postprocess_merge.py — Merge and process multiple raw DAQ files.

For each selected raw file (data_raw_*.txt):
  1. Convert raw voltages → physical units (same formulas as DataProcessing.py)
  2. Tare: average first RAMP_SECONDS of converted values → per-file baseline
  3. Subtract baseline from all rows
  4. Exclude ramp rows from merged output

All files merged in selection order. New time column starts at 0 at 16 Hz.

Output columns:
  time_s              — seconds (0, 1/16, 2/16, ...)
  DCDT_*              — displacement in inches
  *_pressure_psi      — pressure in psi
  SG_*                — strain in microstrains (µε)

Usage:
    python postprocess_merge.py   # opens file browser for multiple selection
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16       # Hz (output sample rate)
RAMP_SECONDS = 5.0      # seconds used for tare baseline and excluded from output
RAMP_ROWS    = int(RAMP_SECONDS * SAMPLE_RATE)   # 80 rows
KPA_TO_PSI   = 0.145038

# V → inches scale per DCDT channel (same order as DataProcessing.py ai0–ai11)
DISP_SCALE = [
    3.937  / 10.0,   # DCDT_Right_Slab_A1   ai0
    3.937  / 10.0,   # DCDT_Right_Slab_A2   ai1
    3.937  / 10.0,   # DCDT_Right_Slab_A3   ai2
    3.937  / 10.0,   # DCDT_Right_Slab_B1   ai3
    1.969  / 10.0,   # DCDT_Right_Slab_B3   ai4  (2 in stroke)
    3.937  / 10.0,   # DCDT_Left_Slab_B1    ai5
    3.937  / 10.0,   # DCDT_Left_Slab_B2_Bot ai6
    1.969  / 10.0,   # DCDT_Left_Slab_B3    ai7  (2 in stroke)
    3.937  / 10.0,   # DCDT_Left_Slab_C1    ai8
    3.937  / 10.0,   # DCDT_Left_Slab_C2    ai9
    3.937  / 10.0,   # DCDT_Left_Slab_C3    ai10
    0.9843 / 10.0,   # DCDT_Beam_B2_Top     ai11 (1 in stroke)
]

DISP_COLS = [
    "DCDT_Right_Slab_A1", "DCDT_Right_Slab_A2", "DCDT_Right_Slab_A3",
    "DCDT_Right_Slab_B1",  "DCDT_Right_Slab_B3",
    "DCDT_Left_Slab_B1",   "DCDT_Left_Slab_B2_Bot", "DCDT_Left_Slab_B3",
    "DCDT_Left_Slab_C1",   "DCDT_Left_Slab_C2",     "DCDT_Left_Slab_C3",
    "DCDT_Beam_B2_Top",
]

PRESS_RAW_COLS = ["volt_ch17", "volt_ch18", "volt_ch19", "volt_ch20"]
PRESS_OUT_COLS = [
    "soil_plate_pressure_psi",
    "agg_plate_pressure_psi",
    "soil_pore_water_pressure_psi",
    "agg_pore_water_pressure_psi",
]

STRAIN_COLS = [
    "SG_2E_top", "SG_3E_top", "SG_4E_top", "SG_4E_bot",
    "SG_5E_top", "SG_5E_bot", "SG_6E_top", "SG_7E_top",
]

ALL_OUT_COLS = ["time_s"] + DISP_COLS + PRESS_OUT_COLS + STRAIN_COLS

# ── Pressure conversion formulas (input in millivolts → output in kPa) ───────
def _soil_plate(mv):  return -2.86e-5 * mv**2 + 1.0038 * mv + 0.9331
def _agg_plate(mv):   return -1.34e-4 * mv**2 + 2.5171 * mv - 1.3375
def _soil_pore(mv):   return -2.79e-5 * mv**2 + 1.0006 * mv + 0.4014
def _agg_pore(mv):    return -2.93e-5 * mv**2 + 1.0073 * mv + 0.9260
PRESS_FUNCS = [_soil_plate, _agg_plate, _soil_pore, _agg_pore]

# ── Process one raw file ──────────────────────────────────────────────────────
def process_file(path):
    """
    Returns:
        data   — ndarray (n_rows - RAMP_ROWS, 24): tared physical values
                 columns: 12 disp (in) + 4 press (psi) + 8 strain (µε)
        tares  — dict with baseline values for logging
    Returns (None, None) if file is too short.
    """
    df = pd.read_csv(path, sep="\t")
    n  = len(df)
    if n <= RAMP_ROWS:
        print(f"  WARNING: {os.path.basename(path)} has only {n} rows "
              f"(need > {RAMP_ROWS}), skipping.")
        return None, None

    # ── DCDT: volts → inches ─────────────────────────────────────────────────
    disp_in = df[DISP_COLS].values * np.array(DISP_SCALE)          # (n, 12) inches

    # ── Pressure: volts → kPa  (formulas expect millivolts) ─────────────────
    press_kpa = np.column_stack([
        PRESS_FUNCS[i](df[PRESS_RAW_COLS[i]].values * 1000)
        for i in range(4)
    ])                                                               # (n, 4) kPa

    # ── Strain: dimensionless from NI-9235 (no conversion needed yet) ────────
    strain = df[STRAIN_COLS].values                                 # (n, 8)

    # ── Tare: mean of first RAMP_ROWS rows (in physical units) ───────────────
    base_disp   = disp_in  [:RAMP_ROWS].mean(axis=0)              # (12,) inches
    base_press  = press_kpa[:RAMP_ROWS].mean(axis=0)              # (4,)  kPa
    base_strain = strain   [:RAMP_ROWS].mean(axis=0)              # (8,)  strain

    # ── Subtract baseline, slice off ramp rows ───────────────────────────────
    disp_tared   = (disp_in   - base_disp  )[RAMP_ROWS:]           # (rows, 12) in
    press_tared  = (press_kpa - base_press )[RAMP_ROWS:]           # (rows, 4)  kPa
    strain_tared = (strain    - base_strain)[RAMP_ROWS:]            # (rows, 8)

    # ── Final unit conversions ────────────────────────────────────────────────
    press_psi = press_tared * KPA_TO_PSI                           # psi
    strain_us = strain_tared * 1e6                                  # microstrains

    tares = {
        "disp_in_baseline":   base_disp,
        "press_kpa_baseline": base_press,
        "strain_baseline":    base_strain,
    }
    return np.hstack([disp_tared, press_psi, strain_us]), tares    # (rows, 24)

# ── File selection ────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.withdraw()

in_paths = filedialog.askopenfilenames(
    title="Select trimmed files to merge (*_trimmed.txt)",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Trimmed files", "*_trimmed.txt"),
               ("Text files",    "*.txt"),
               ("All files",     "*.*")]
)

if not in_paths:
    messagebox.showinfo("Cancelled", "No files selected. Exiting.")
    root.destroy()
    raise SystemExit(0)

root.destroy()

in_paths = sorted(in_paths)
print(f"\nSelected {len(in_paths)} file(s):")
for p in in_paths:
    print(f"  {os.path.basename(p)}")

# ── Process and merge ─────────────────────────────────────────────────────────
chunks         = []
file_row_counts = []

for p in in_paths:
    print(f"\nProcessing: {os.path.basename(p)}")
    chunk, tares = process_file(p)
    if chunk is None:
        continue
    chunks.append(chunk)
    file_row_counts.append(chunk.shape[0])
    n_rows = chunk.shape[0]
    print(f"  Rows (after ramp exclusion) : {n_rows}  ({n_rows / SAMPLE_RATE:.2f} s)")
    print(f"  DCDT baseline (in)          : {['%+.5f' % v for v in tares['disp_in_baseline']]}")
    print(f"  Pressure baseline (kPa)     : {['%+.4f' % v for v in tares['press_kpa_baseline']]}")
    print(f"  Strain baseline (µε)        : {['%+.2f' % (v*1e6) for v in tares['strain_baseline']]}")

if not chunks:
    print("\nNo valid files processed. Exiting.")
    raise SystemExit(1)

merged  = np.vstack(chunks)
n_total = merged.shape[0]
time_col = np.arange(n_total) / SAMPLE_RATE

# Cumulative row boundaries for file-boundary markers in the plot
cum_rows = np.cumsum([0] + file_row_counts)
boundary_times = [cum_rows[i] / SAMPLE_RATE for i in range(1, len(cum_rows) - 1)]

print(f"\n{'─'*55}")
print(f"  Files merged   : {len(chunks)}")
print(f"  Total rows     : {n_total:,}")
print(f"  Total duration : {n_total / SAMPLE_RATE:.2f} s  "
      f"({n_total / SAMPLE_RATE / 60:.1f} min)")
print(f"{'─'*55}")

# ── Write output file ─────────────────────────────────────────────────────────
out_dir  = os.path.dirname(in_paths[0])
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(out_dir, f"merged_processed_{ts}.txt")

print(f"\nWriting: {out_path}")
with open(out_path, "w") as f:
    f.write("\t".join(ALL_OUT_COLS) + "\n")
    for i in range(n_total):
        row = [f"{time_col[i]:.6f}"] + [f"{v:.6f}" for v in merged[i]]
        f.write("\t".join(row) + "\n")
print(f"Done — {n_total} rows written.")
print(f"  Columns: {len(DISP_COLS)} displacement (in)  |  "
      f"{len(PRESS_OUT_COLS)} pressure (psi)  |  "
      f"{len(STRAIN_COLS)} strain (µε)")

# ── Interactive viewer ────────────────────────────────────────────────────────
WINDOW_S = 20.0

def ch_color(c):
    if c.startswith("DCDT_"):     return "steelblue"
    if c.startswith("SG_"):       return "tomato"
    if "pressure" in c.lower():   return "darkorange"
    return "gray"

data_cols = DISP_COLS + PRESS_OUT_COLS + STRAIN_COLS
colors    = [ch_color(c) for c in data_cols]
vis       = [c.startswith("DCDT_") for c in data_cols]
if not any(vis):
    vis[0] = True

leg_line_map = {}

fig = plt.figure(figsize=(17, 9))
fig.suptitle(
    f"Merged processed data  —  {len(chunks)} file(s)  |  "
    f"{n_total} rows  |  {n_total / SAMPLE_RATE:.1f} s total\n"
    "Scroll slider to navigate  •  Click legend to toggle channel  •  "
    "Group buttons to switch view  •  Dashed black lines = file boundaries",
    fontsize=8.5, fontweight="bold"
)

plot_ax  = fig.add_axes([0.08, 0.13, 0.90, 0.79])
slide_ax = fig.add_axes([0.08, 0.04, 0.90, 0.04])

bw, bh, bx = 0.065, 0.048, 0.005
btn_dcdt  = Button(fig.add_axes([bx, 0.80, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
btn_sg    = Button(fig.add_axes([bx, 0.75, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
btn_press = Button(fig.add_axes([bx, 0.70, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
btn_all   = Button(fig.add_axes([bx, 0.65, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
btn_none  = Button(fig.add_axes([bx, 0.60, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none):
    btn.label.set_fontsize(7)

t_max_sl = max(float(time_col[-1]) - WINDOW_S, 0.01)
slider   = Slider(slide_ax, "Time (s)", 0.0, t_max_sl,
                  valinit=0.0, color="steelblue")

def redraw(_=None, keep_xlim=False):
    saved_xlim = plot_ax.get_xlim() if keep_xlim else None
    t_left  = slider.val
    t_right = t_left + WINDOW_S
    mask    = (time_col >= t_left) & (time_col <= t_right)
    win_t   = time_col[mask]

    plot_ax.cla()
    leg_line_map.clear()

    plotted      = []
    y_label_set  = []
    for i, c in enumerate(data_cols):
        if vis[i]:
            plot_ax.plot(win_t, merged[mask, i],
                         lw=0.9, color=colors[i], label=c, alpha=0.85)
            plotted.append(c)
            if c.startswith("DCDT_") and "Displacement (in)" not in y_label_set:
                y_label_set.append("Displacement (in)")
            elif "pressure" in c.lower() and "Pressure (psi)" not in y_label_set:
                y_label_set.append("Pressure (psi)")
            elif c.startswith("SG_") and "Strain (µε)" not in y_label_set:
                y_label_set.append("Strain (µε)")

    # File boundary markers
    for t_b in boundary_times:
        if t_left - 1 <= t_b <= t_right + 1:
            plot_ax.axvline(t_b, color="black", lw=1.2, ls="--", alpha=0.5,
                            label=f"boundary ({t_b:.1f} s)")

    plot_ax.set_title("Merged Processed Data — zoom/scroll freely", fontsize=9)
    if keep_xlim and saved_xlim is not None:
        plot_ax.set_xlim(saved_xlim)
    else:
        plot_ax.set_xlim(t_left, t_right)
    plot_ax.set_xlabel("Time (s)", fontsize=9)
    plot_ax.set_ylabel("  /  ".join(y_label_set) if y_label_set else "Value", fontsize=8)
    plot_ax.grid(True, alpha=0.3)

    if plotted:
        leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
        for ll, ch_name in zip(leg.get_lines(), plotted):
            ll.set_picker(8)
            ll.set_linewidth(2.5)
            leg_line_map[ll] = ch_name

    fig.canvas.draw_idle()

def set_group(group):
    for i, c in enumerate(data_cols):
        if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
        elif group == "sg":    vis[i] = c.startswith("SG_")
        elif group == "press": vis[i] = "pressure" in c.lower()
        elif group == "all":   vis[i] = True
        elif group == "none":  vis[i] = False
    redraw(keep_xlim=True)

def on_legend_pick(event):
    ll = event.artist
    if ll not in leg_line_map:
        return
    idx = data_cols.index(leg_line_map[ll])
    vis[idx] = not vis[idx]
    redraw(keep_xlim=True)

slider.on_changed(redraw)
btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
btn_sg.on_clicked(lambda _:    set_group("sg"))
btn_press.on_clicked(lambda _: set_group("press"))
btn_all.on_clicked(lambda _:   set_group("all"))
btn_none.on_clicked(lambda _:  set_group("none"))
fig.canvas.mpl_connect("pick_event", on_legend_pick)

redraw()
plt.show()
