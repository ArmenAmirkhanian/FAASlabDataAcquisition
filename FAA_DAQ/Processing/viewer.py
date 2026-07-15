"""
data_viewer.py — Quick interactive viewer for any DAQ txt file.

Opens a tab-separated file and plots all channels interactively.
No trimming, no markers — just data exploration.

Usage:
    python data_viewer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

WINDOW_S  = 15.0   # seconds visible in the plot window at once
MARKER_MS = 3.5
LINE_W    = 1.2

# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

in_path = filedialog.askopenfilename(
    title="Select data file to view",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

if not in_path:
    messagebox.showinfo("Cancelled", "No file selected. Exiting.")
    root.destroy()
    raise SystemExit(0)

root.destroy()

# ── Load ──────────────────────────────────────────────────────────────────────
#df = pd.read_csv(in_path, sep="\t"), for python txt files
df = pd.read_csv(in_path, sep="\t", skiprows=list(range(6)) + [7])  # MTS: 6 metadata lines, header, units row

if "time_s" in df.columns:
    time = df["time_s"].values
    cols = [c for c in df.columns if c != "time_s"]
else:
    time = np.arange(len(df))
    cols = list(df.columns)

data = df[cols].values.astype(float)
N    = len(time)

print(f"File     : {os.path.basename(in_path)}")
print(f"Rows     : {N:,}")
print(f"Channels : {len(cols)}")
print(f"Duration : {time[-1]:.2f} s  ({time[-1]/60:.1f} min)")

# ── Channel color and group detection ────────────────────────────────────────
_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39",
]
colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cols))]
vis    = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
if not any(vis):
    vis = [True] + [False] * (len(cols) - 1)

leg_line_map = {}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9))
fig.suptitle(
    f"{os.path.basename(in_path)}  —  {N:,} rows  |  {time[-1]:.1f} s total\n"
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
btn_next = Button(fig.add_axes([bx, 0.55, bw, bh]), f"Next {WINDOW_S:g}s", color="#dde8ff", hovercolor="#bbd0ff")
btn_prev = Button(fig.add_axes([bx, 0.50, bw, bh]), f"Prev {WINDOW_S:g}s", color="#dde8ff", hovercolor="#bbd0ff")
for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_next, btn_prev):
    btn.label.set_fontsize(7)

t_max_sl = max(float(time[-1]) - WINDOW_S, 0.01)
slider   = Slider(slide_ax, "Time (s)", 0.0, t_max_sl, valinit=0.0, color="steelblue")

# ── Draw ──────────────────────────────────────────────────────────────────────
def redraw(_=None, keep_xlim=False):
    saved_xlim = plot_ax.get_xlim() if keep_xlim else None
    t_left  = slider.val
    t_right = t_left + WINDOW_S
    mask    = (time >= t_left) & (time <= t_right)
    win_t   = time[mask]

    plot_ax.cla()
    leg_line_map.clear()

    plotted    = []
    y_labels   = []
    for i, c in enumerate(cols):
        if vis[i]:
            plot_ax.plot(win_t, data[mask, i],
                         lw=LINE_W, color=colors[i], label=c, alpha=0.85,
                         marker="o", markersize=MARKER_MS)
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

def do_next(_): slider.set_val(min(slider.val + WINDOW_S, t_max_sl))
def do_prev(_): slider.set_val(max(slider.val - WINDOW_S, 0.0))

slider.on_changed(redraw)
btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
btn_sg.on_clicked(lambda _:    set_group("sg"))
btn_press.on_clicked(lambda _: set_group("press"))
btn_all.on_clicked(lambda _:   set_group("all"))
btn_none.on_clicked(lambda _:  set_group("none"))
btn_next.on_clicked(do_next)
btn_prev.on_clicked(do_prev)
fig.canvas.mpl_connect("pick_event", on_legend_pick)

redraw()
plt.show()
