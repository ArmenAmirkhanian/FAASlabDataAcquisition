"""
postprocess_merge.py — Simple multi-file merger with interactive trim selection.

For each selected file (any filename):
  1. Load it as-is — no unit conversion, no taring.
  2. Skip the first RAMP_SECONDS of every file automatically (pre-load /
     ramp) — this is the start of both the voltage and strain ranges, no
     marker needed.
  3. Show an interactive plot of all its columns. Arm "Voltage End" or
     "Strain End" and click on the plot to place each marker, then click
     "Confirm". Voltage (DCDT_/pressure/volt_ch) and strain (SG_) are
     trimmed INDEPENDENTLY, since strain cycles can lag voltage cycles.
  4. Crop each group from the end of its ramp to its own marked end.

Across files, the voltage-group rows are concatenated end-to-end in
selection order, and the strain-group rows are concatenated end-to-end
separately — so the first cycle of file N+1 is appended right after the
last cycle of file N, for each group independently. The two groups can end
up different lengths after merging; the shorter one is padded with blank
rows at the end. No processing is applied to the data values themselves —
time_s (if present) is renumbered to stay continuous across the merged rows.

Usage:
    python postprocess_merge.py
"""

import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

SAMPLE_RATE  = 16   # Hz — only used to regenerate a continuous time_s column after merging
CYCLE_HZ     = 1.0  # nominal cyclic-loading frequency, Hz
RAMP_SECONDS = 5.0   # skip this much at the start of every file (pre-load / ramp)
MARKER_MS    = 3.5
LINE_W       = 1.0
# samples averaged at each boundary when matching continuity — several full cycles
# (median, not mean) so a single noisy/partial cycle at the trim boundary can't
# skew the estimate and bake a permanent offset error into the rest of the segment
MATCH_CYCLES  = 10
MATCH_SAMPLES = int(round(SAMPLE_RATE / CYCLE_HZ)) * MATCH_CYCLES
OFFSET_FLAG_THRESHOLD = 0.01  # print per-column detail if |offset| exceeds this

# 24-colour palette (same as process_badcycles_fix.py / postprocess_peaks.py)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]


def is_voltage_col(c):
    return c.startswith("DCDT_") or "pressure" in c.lower() or c.startswith("volt_ch")


def is_strain_col(c):
    return c.startswith("SG_")


def pick_trim_ranges(df, filename):
    """Interactive marker placement: arm Voltage End or Strain End
    independently, click on the plot to place each one, then Confirm. Same
    interaction style as process_trim.py's drop-region selector, but used
    here to trim the two groups to separate ranges (voltage and strain
    cycles can be out of phase). The start of both ranges is fixed at
    RAMP_SECONDS into the file — no marker needed.

    Returns (v_start_idx, v_end_idx, s_start_idx, s_end_idx).
    """
    WINDOW_S = 35.0

    if "time_s" in df.columns:
        time_arr = df["time_s"].to_numpy()
    else:
        time_arr = np.arange(len(df), dtype=float)
    data_cols = [c for c in df.columns if c != "time_s"]
    data_arr = df[data_cols].to_numpy(float)

    ramp_rows = int(round(RAMP_SECONDS * SAMPLE_RATE))
    ramp_end_t = float(time_arr[min(ramp_rows, len(time_arr) - 1)])

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(data_cols))]
    vis = [c.startswith("DCDT_") for c in data_cols]
    if not any(vis):
        vis[0] = True

    MODE_INFO = {
        "v_end":   ("Voltage\nEnd",   "#cc6600", "#ffd090"),
        "s_end":   ("Strain\nEnd",    "#7700cc", "#e0b0ff"),
    }
    marks        = {k: [None] for k in MODE_INFO}
    mode         = [None]
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        f"{filename}\n"
        f"First {RAMP_SECONDS:g}s (ramp) auto-excluded from both groups — dotted grey line  |  "
        "Click a marker button to arm it -> click on plot to place  |  "
        "Click armed button again to disarm (zoom/pan freely)\n"
        "Voltage and Strain are trimmed independently  •  Scroll to navigate  •  "
        "Click legend to toggle channel  •  Click Confirm when done",
        fontsize=8.5, fontweight="bold"
    )

    plot_ax  = fig.add_axes([0.14, 0.13, 0.84, 0.79])
    slide_ax = fig.add_axes([0.14, 0.04, 0.84, 0.04])

    bw, bh, bx = 0.065, 0.048, 0.005
    btn_dcdt  = Button(fig.add_axes([bx, 0.85, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.80, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.75, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.70, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.65, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

    mode_axes, mode_btns = {}, {}
    y = 0.56
    for key in ("v_end", "s_end"):
        label, _armed_col, idle_col = MODE_INFO[key]
        ax_  = fig.add_axes([bx, y, bw, bh])
        btn_ = Button(ax_, label, color=idle_col, hovercolor=idle_col)
        mode_axes[key] = ax_
        mode_btns[key] = btn_
        y -= 0.05

    btn_confirm = Button(fig.add_axes([bx, y - 0.03, bw, bh]), "Confirm", color="#90ee90", hovercolor="#60dd60")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_confirm, *mode_btns.values()):
        btn.label.set_fontsize(7)

    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                    valinit=float(time_arr[0]), color="steelblue")

    def update_mode_buttons():
        for key, ax_ in mode_axes.items():
            _label, armed_col, idle_col = MODE_INFO[key]
            armed = (mode[0] == key)
            c = armed_col if armed else idle_col
            mode_btns[key].color      = c
            mode_btns[key].hovercolor = c
            mode_btns[key].label.set_color("white" if armed else "black")
            ax_.set_facecolor(c)
        fig.canvas.draw_idle()

    update_mode_buttons()

    def redraw(_=None, keep_xlim=False):
        saved_xlim = plot_ax.get_xlim() if keep_xlim else None
        t_left  = slider.val
        t_right = t_left + WINDOW_S
        mask    = (time_arr >= t_left) & (time_arr <= t_right)
        win_t   = time_arr[mask]

        plot_ax.cla()
        leg_line_map.clear()

        # always plot every channel (so it always has a legend entry to click back
        # on) — toggled-off channels are just made invisible, not omitted
        for i, c in enumerate(data_cols):
            line, = plot_ax.plot(win_t, data_arr[mask, i],
                                 lw=LINE_W, color=colors[i], label=c, alpha=0.85,
                                 marker="o", markersize=MARKER_MS)
            line.set_visible(vis[i])
        plot_ax.relim(visible_only=True)
        plot_ax.autoscale_view()

        if t_left <= ramp_end_t <= t_right:
            plot_ax.axvline(ramp_end_t, color="gray", lw=1.5, ls=":", zorder=9,
                            label=f"Ramp end (auto): {ramp_end_t:.4f} s")

        for key, (label, col, _idle) in MODE_INFO.items():
            t = marks[key][0]
            if t is not None:
                flat_label = label.replace("\n", " ")
                plot_ax.axvline(t, color=col, lw=2.0, ls="-", zorder=10,
                                label=f"{flat_label}: {t:.4f} s")

        if mode[0] is not None:
            active_col  = MODE_INFO[mode[0]][1]
            flat_label  = MODE_INFO[mode[0]][0].replace("\n", " ").upper()
            title_line1 = f"ARMED: {flat_label} — click on plot to place"
        else:
            active_col  = "gray"
            title_line1 = "No marker armed — zoom/pan freely"
        plot_ax.set_title(
            f"{title_line1}\nClick a legend entry to toggle that channel",
            fontsize=8.5, color=active_col
        )

        if keep_xlim and saved_xlim is not None:
            plot_ax.set_xlim(saved_xlim)
        else:
            plot_ax.set_xlim(t_left, t_right)
        plot_ax.set_xlabel("time_s" if "time_s" in df.columns else "row index", fontsize=9)
        plot_ax.set_ylabel("Value", fontsize=9)
        plot_ax.grid(True, alpha=0.3)

        leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
        for leg_line, ch_name in zip(leg.get_lines(), data_cols):
            leg_line.set_picker(8)
            leg_line.set_linewidth(2.5)
            leg_line.set_alpha(1.0 if vis[data_cols.index(ch_name)] else 0.25)
            leg_line_map[leg_line] = ch_name

        fig.canvas.draw_idle()

    def set_group(group):
        for i, c in enumerate(data_cols):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw(keep_xlim=True)

    def toggle_mode(m):
        mode[0] = None if mode[0] == m else m
        update_mode_buttons()
        redraw(keep_xlim=True)

    def on_legend_pick(event):
        leg_line = event.artist
        if leg_line not in leg_line_map:
            return
        idx = data_cols.index(leg_line_map[leg_line])
        vis[idx] = not vis[idx]
        redraw(keep_xlim=True)

    def on_plot_click(event):
        if mode[0] is None:
            return
        if event.inaxes is not plot_ax or event.button != 1 or event.xdata is None:
            return
        leg = plot_ax.get_legend()
        if leg is not None and leg.get_window_extent().contains(event.x, event.y):
            return
        snapped = float(time_arr[int(np.argmin(np.abs(time_arr - event.xdata)))])
        marks[mode[0]][0] = snapped
        redraw(keep_xlim=True)

    def on_confirm(_):
        plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
    btn_sg.on_clicked(lambda _:    set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _:   set_group("all"))
    btn_none.on_clicked(lambda _:  set_group("none"))
    for key in mode_btns:
        mode_btns[key].on_clicked(lambda _, k=key: toggle_mode(k))
    btn_confirm.on_clicked(on_confirm)
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    def to_idx(t, default):
        if t is None:
            return default
        return int(np.argmin(np.abs(time_arr - t)))

    v0 = s0 = min(ramp_rows, len(df) - 1)
    v1 = to_idx(marks["v_end"][0], len(df) - 1)
    s1 = to_idx(marks["s_end"][0], len(df) - 1)
    if v0 > v1:
        v0, v1 = v1, v0
    if s0 > s1:
        s0, s1 = s1, s0
    return v0, v1, s0, s1


def natural_key(path):
    """Sort key that orders embedded numbers numerically (Set_2 before Set_10)."""
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def main():
    root = tk.Tk()
    root.withdraw()

    in_paths = filedialog.askopenfilenames(
        title="Select files to merge (any filename)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not in_paths:
        messagebox.showinfo("Cancelled", "No files selected. Exiting.")
        root.destroy()
        return

    in_paths = sorted(in_paths, key=natural_key)
    print(f"\nSelected {len(in_paths)} file(s):")
    for p in in_paths:
        print(f"  {os.path.basename(p)}")

    v_chunks, s_chunks = [], []
    voltage_cols = strain_cols = None
    had_time_col = False
    last_v_vals = last_s_vals = None   # per-column boundary value from the previous chunk

    def match_continuity(chunk, last_vals, label):
        """Shift each column by a constant so its start matches where the
        previous chunk's same column left off (median of MATCH_SAMPLES —
        several cycles — at each end, robust to a noisy/partial cycle right
        at the trim boundary) — removes the between-file baseline jump/drift."""
        if last_vals is None or chunk.empty:
            new_last = ({c: chunk[c].iloc[-MATCH_SAMPLES:].median() for c in chunk.columns}
                        if not chunk.empty else last_vals)
            return chunk, new_last
        n = min(MATCH_SAMPLES, len(chunk))
        shifted = chunk.copy()
        offsets = {}
        for c in chunk.columns:
            offset = last_vals[c] - chunk[c].iloc[:n].median()
            shifted[c] = chunk[c] + offset
            offsets[c] = offset
        if offsets:
            vals = list(offsets.values())
            print(f"  {label} continuity offsets: {min(vals):+.5f} to {max(vals):+.5f}")
            big = {c: o for c, o in offsets.items() if abs(o) > OFFSET_FLAG_THRESHOLD}
            if big:
                detail = ", ".join(f"{c}={o:+.4f}" for c, o in
                                    sorted(big.items(), key=lambda kv: -abs(kv[1])))
                print(f"    >{OFFSET_FLAG_THRESHOLD:g} on: {detail}")
        new_last = {c: shifted[c].iloc[-MATCH_SAMPLES:].median() for c in shifted.columns}
        return shifted, new_last

    for p in in_paths:
        df = pd.read_csv(p, sep="\t")
        if voltage_cols is None:
            had_time_col = "time_s" in df.columns
            all_cols = [c for c in df.columns if c != "time_s"]
            voltage_cols = [c for c in all_cols if is_voltage_col(c)]
            strain_cols  = [c for c in all_cols if is_strain_col(c)]
            other_cols   = [c for c in all_cols if c not in voltage_cols and c not in strain_cols]
            if other_cols:
                print(f"  Note: unclassified columns {other_cols} will be trimmed with the voltage group.")
                voltage_cols = voltage_cols + other_cols

        print(f"\n{os.path.basename(p)}: {len(df)} rows — select ranges to keep...")
        v0, v1, s0, s1 = pick_trim_ranges(df, os.path.basename(p))
        v_chunk = df[voltage_cols].iloc[v0:v1 + 1].reset_index(drop=True)
        s_chunk = df[strain_cols].iloc[s0:s1 + 1].reset_index(drop=True)
        print(f"  Voltage rows {v0}-{v1}  ({len(v_chunk)} rows)   "
              f"Strain rows {s0}-{s1}  ({len(s_chunk)} rows)")

        v_chunk, last_v_vals = match_continuity(v_chunk, last_v_vals, "Voltage")
        s_chunk, last_s_vals = match_continuity(s_chunk, last_s_vals, "Strain")

        v_chunks.append(v_chunk)
        s_chunks.append(s_chunk)

    if not v_chunks:
        print("No files processed. Exiting.")
        root.destroy()
        return

    v_merged = pd.concat(v_chunks, ignore_index=True)
    s_merged = pd.concat(s_chunks, ignore_index=True)
    n_v, n_s = len(v_merged), len(s_merged)
    n_total = max(n_v, n_s)
    if n_v != n_s:
        print(f"\nNote: voltage group has {n_v} rows, strain group has {n_s} rows after merging "
              f"— the shorter group is padded with blank rows at the end.")

    def pad_to(series, n):
        arr = series.to_numpy(dtype=float)
        if len(arr) >= n:
            return arr[:n]
        return np.concatenate([arr, np.full(n - len(arr), np.nan)])

    merged = pd.DataFrame()
    if had_time_col:
        merged["time_s"] = np.arange(n_total) / SAMPLE_RATE
    for c in voltage_cols:
        merged[c] = pad_to(v_merged[c], n_total)
    for c in strain_cols:
        merged[c] = pad_to(s_merged[c], n_total)

    out_path = filedialog.asksaveasfilename(
        title="Save merged file as",
        defaultextension=".txt",
        initialfile=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()

    if not out_path:
        print("No save location selected. Exiting.")
        return

    merged.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nDone — {len(merged)} rows written to:\n  {out_path}")


if __name__ == "__main__":
    main()
