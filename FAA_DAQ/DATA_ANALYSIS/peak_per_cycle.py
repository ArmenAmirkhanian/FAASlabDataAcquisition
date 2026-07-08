"""
peak_per_cycle.py — Extract one peak value per cycle for every channel.

Uses upward zero-crossings of the first DCDT channel to define cycle
boundaries (robust to phase drift and dropouts).  Within each cycle window,
records the maximum (high peak) value of every channel and the time at
which that peak occurs.

Output columns:
  cycle      — cycle index (1-based)
  t_start_s  — time of the cycle start (upward crossing)
  t_peak_s   — time of the high peak in the reference DCDT channel
  n_rows     — number of rows in the cycle (< SPC-3 = compressed/bad cycle)
  <ch_name>  — peak value of that channel within the cycle  (one col per ch)

Usage:
    python peak_per_cycle.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16     # Hz
CYCLE_HZ     = 1.0    # Hz  (loading frequency)
RAMP_SECONDS = 5.0    # seconds to skip at the start (non-cyclic ramp)

# ── Derived ───────────────────────────────────────────────────────────────────
SPC       = int(round(SAMPLE_RATE / CYCLE_HZ))   # rows per cycle = 16
RAMP_ROWS = int(RAMP_SECONDS * SAMPLE_RATE)       # rows to skip   = 80

# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
in_path = filedialog.askopenfilename(
    title="Select data file",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)
if not in_path:
    messagebox.showinfo("Cancelled", "No file selected.")
    root.destroy()
    raise SystemExit(0)
root.destroy()

# ── Load ──────────────────────────────────────────────────────────────────────
df   = pd.read_csv(in_path, sep="\t")
cols = [c for c in df.columns if c != "time_s"]
data = df[cols].values.astype(float)
N    = len(data)

time = df["time_s"].values if "time_s" in df.columns else np.arange(N) / SAMPLE_RATE

print(f"\nFile     : {os.path.basename(in_path)}")
print(f"Rows     : {N:,}  ({time[-1]:.2f} s)")
print(f"Channels : {len(cols)}")

# ── Choose reference channel (first DCDT) ─────────────────────────────────────
dcdt_idx = [i for i, c in enumerate(cols) if c.startswith("DCDT_")]
ref_idx  = dcdt_idx[0] if dcdt_idx else 0
print(f"Reference: {cols[ref_idx]}  (used for cycle boundary detection)")

# ── Upward zero-crossing detection ────────────────────────────────────────────
ref_cyc = data[RAMP_ROWS:, ref_idx]
level   = (float(np.percentile(ref_cyc, 5)) + float(np.percentile(ref_cyc, 95))) / 2.0

raw_xings = []
for i in range(len(ref_cyc) - 1):
    if ref_cyc[i] < level <= ref_cyc[i + 1]:
        frac = (level - ref_cyc[i]) / (ref_cyc[i + 1] - ref_cyc[i])
        raw_xings.append(i + frac)

# Shift back to full-array row indices
xings = np.array(raw_xings) + RAMP_ROWS
print(f"Crossings: {len(xings):,}  (crossing level = {level:.4f})")

# ── Extract peak per cycle ─────────────────────────────────────────────────────
records = []

for k in range(len(xings) - 1):
    i0 = int(np.floor(xings[k]))
    i1 = int(np.ceil(xings[k + 1]))
    i1 = min(i1, N)

    if i1 <= i0:
        continue

    window  = data[i0:i1, :]          # shape (n_rows, n_ch)
    n_rows  = i1 - i0
    t_start = float(time[i0])

    # Peak index within the window for the reference channel
    ref_peak_local = int(np.argmax(window[:, ref_idx]))
    t_peak = float(time[i0 + ref_peak_local])

    # Peak value of every channel within this cycle window
    peak_vals = window.max(axis=0).tolist()

    records.append([k + 1, t_start, t_peak, n_rows] + peak_vals)

# ── Build output DataFrame ────────────────────────────────────────────────────
out_cols = ["cycle", "t_start_s", "t_peak_s", "n_rows"] + cols
out_df   = pd.DataFrame(records, columns=out_cols)

n_compressed = int((out_df["n_rows"] < SPC - 3).sum())
print(f"\nCycles extracted : {len(out_df):,}")
print(f"Compressed cycles: {n_compressed:,}  "
      f"(n_rows < {SPC - 3}, likely bad — dropouts not yet fixed)")

# ── Save ──────────────────────────────────────────────────────────────────────
default_name = os.path.splitext(os.path.basename(in_path))[0] + "_peaks.txt"

save_root = tk.Tk()
save_root.withdraw()
out_path = filedialog.asksaveasfilename(
    title="Save peak file as...",
    initialdir=os.path.dirname(in_path),
    initialfile=default_name,
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)
save_root.destroy()

if not out_path:
    print("Save cancelled.")
    raise SystemExit(0)

out_path = os.path.normpath(out_path)
save_dir = os.path.dirname(out_path)
os.makedirs(save_dir, exist_ok=True)
out_df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")

print(f"\nSaved  : {out_path}")
print(f"Columns: cycle, t_start_s, t_peak_s, n_rows, "
      f"then one peak column per channel")

# ── Total change table (final peak − initial peak) ───────────────────────────
summary_cols = [c for c in cols
                if c.startswith("DCDT_") or c.startswith("SG_")]

first_row = out_df.iloc[0]
last_row  = out_df.iloc[-1]

table_data = [[ch,
               f"{float(first_row[ch]):.5f}",
               f"{float(last_row[ch]):.5f}",
               f"{float(last_row[ch]) - float(first_row[ch]):.5f}"]
              for ch in summary_cols]

fig_t, ax_t = plt.subplots(figsize=(9, max(3, len(summary_cols) * 0.35 + 1.5)))
ax_t.axis("off")
ax_t.set_title(
    f"{os.path.basename(in_path)}  —  Total change in peak  (final − initial)",
    fontsize=9, fontweight="bold", pad=10
)

tbl = ax_t.table(
    cellText=table_data,
    colLabels=["Channel", "Initial peak", "Final peak", "Δ (final − initial)"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width([0, 1, 2, 3])

for j in range(4):
    tbl[(0, j)].set_facecolor("#2c5f8a")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

for i in range(1, len(summary_cols) + 1):
    fc = "#f0f4f8" if i % 2 == 0 else "white"
    for j in range(4):
        tbl[(i, j)].set_facecolor(fc)

plt.tight_layout()
plt.show()

# ── Save individual channel plots ─────────────────────────────────────────────
plots_dir = os.path.join(save_dir, "channel_plots")
os.makedirs(plots_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(in_path))[0]

IQR_FACTOR = 3.0   # raise to keep more data, lower to remove more outliers

plot_df = out_df[out_df["n_rows"].values >= (SPC - 3)].reset_index(drop=True).copy()
for ch in cols:
    v  = plot_df[ch]
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    plot_df.loc[(v < q1 - IQR_FACTOR * iqr) | (v > q3 + IQR_FACTOR * iqr), ch] = np.nan

t_plot = plot_df["t_peak_s"].values

def ch_color(c):
    if c.startswith("DCDT_"):                               return "steelblue"
    if c.startswith("SG_"):                                 return "tomato"
    if "pressure" in c.lower() or c.startswith("volt_ch"): return "darkorange"
    return "gray"

print(f"\nSaving individual channel plots to: {plots_dir}")
for ch in cols:
    fig_ch, ax_ch = plt.subplots(figsize=(12, 4))
    ax_ch.plot(t_plot, plot_df[ch].values, lw=0.8, color=ch_color(ch), alpha=0.9)
    ax_ch.set_xlabel("Time (s)", fontsize=9)
    ax_ch.set_ylabel(ch, fontsize=9)
    ax_ch.set_title(f"{base_name}  —  {ch}  (peak per cycle)", fontsize=9)
    ax_ch.grid(True, alpha=0.3)
    fig_ch.tight_layout()
    out_fig = os.path.join(plots_dir, f"{base_name}_{ch}.png")
    fig_ch.savefig(out_fig, dpi=150)
    plt.close(fig_ch)
    print(f"  Saved: {ch}.png")

print(f"Done — {len(cols)} plots saved.")

# ── Interactive peak viewer ───────────────────────────────────────────────────
from matplotlib.widgets import Slider, Button

WINDOW_S = 200.0   # seconds visible at once

# plot_df and outlier filtering already applied in the individual-plots section above
t_all      = plot_df["t_peak_s"].values
n_ch       = len(cols)
_palette   = ([plt.colormaps["tab20"](i)  for i in range(20)] +
              [plt.colormaps["tab20b"](i) for i in range(20)])
colors     = _palette[:n_ch]
vis        = [True] * n_ch

leg_line_map = {}

fig = plt.figure(figsize=(17, 9))
fig.suptitle(
    f"{os.path.basename(in_path)}  —  peak per cycle\n"
    "Scroll slider to navigate  •  Click legend to toggle channel  •  "
    "Group buttons to switch view",
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

t_max_sl = max(float(t_all[-1]) - WINDOW_S, 0.01)
slider   = Slider(slide_ax, "Time (s)", 0.0, t_max_sl, valinit=0.0, color="steelblue")

def redraw(_=None, keep_xlim=False):
    saved_xlim = plot_ax.get_xlim() if keep_xlim else None
    t_left  = slider.val
    t_right = t_left + WINDOW_S
    mask    = (t_all >= t_left) & (t_all <= t_right)
    win_t   = t_all[mask]

    plot_ax.cla()
    leg_line_map.clear()

    plotted  = []
    y_labels = []
    for i, c in enumerate(cols):
        if vis[i]:
            plot_ax.plot(win_t, plot_df[c].values[mask],
                         lw=0.9, color=colors[i], label=c, alpha=0.85,
                         marker=".", markersize=3)
            plotted.append(c)
            if c.startswith("DCDT_") and "Displacement" not in y_labels:
                y_labels.append("Displacement")
            elif ("pressure" in c.lower() or c.startswith("volt_ch")) and "Pressure / Voltage" not in y_labels:
                y_labels.append("Pressure / Voltage")
            elif c.startswith("SG_") and "Strain" not in y_labels:
                y_labels.append("Strain")

    plot_ax.set_xlabel("Time (s)", fontsize=9)
    plot_ax.set_ylabel("  /  ".join(y_labels) if y_labels else "Value", fontsize=8)
    plot_ax.set_title(os.path.basename(in_path), fontsize=9)
    if keep_xlim and saved_xlim is not None:
        plot_ax.set_xlim(saved_xlim)
    else:
        plot_ax.set_xlim(t_left, t_right)
    plot_ax.grid(True, alpha=0.3)

    if plotted:
        leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
        for ll, ch_name in zip(leg.get_lines(), plotted):
            ll.set_picker(8)
            ll.set_linewidth(2.5)
            leg_line_map[ll] = ch_name

    fig.canvas.draw_idle()

def set_group(group):
    for i, c in enumerate(cols):
        if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
        elif group == "sg":    vis[i] = c.startswith("SG_")
        elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
        elif group == "all":   vis[i] = True
        elif group == "none":  vis[i] = False
    redraw(keep_xlim=True)

def on_legend_pick(event):
    ll = event.artist
    if ll not in leg_line_map:
        return
    idx = cols.index(leg_line_map[ll])
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
