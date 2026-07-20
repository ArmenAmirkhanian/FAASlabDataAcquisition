#!/usr/bin/env python3
"""
extract_mts_channels.py — Pull selected channels out of an MTS station data
export (e.g. "DAQ- Running Time, ... - (Timed).txt").

FILE LAYOUT HANDLED
    lines 1-6 : file description (path, test name, run, date, blanks)
    line 7    : column header (tab-separated, names have trailing spaces)
    line 8    : units row (sec / kip / in / cycles ...)
    line 9+   : data, tab-separated

WHAT IT EXTRACTS (edit WANTED below to change)
    Running Time (sec)          -> column  time_s
    244.23A Force (kip)         -> column  force_kip
    244.23A Displacement (in)   -> column  displacement_in

A file picker opens for the input file and a folder picker for the output
location. The output is a tab-separated .txt named <input>_extracted.txt
with one header line, ready for pandas / Excel / the peak-extraction script.

USAGE
    python extract_mts_channels.py                 -> pickers for input+output
    python extract_mts_channels.py in.txt outdir   -> no dialogs (scriptable)

REQUIREMENTS: pandas, numpy (tkinter ships with Python).
"""

import os
import sys

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
N_DESC_LINES = 6        # description lines before the header line
UNITS_LINE   = 7        # 0-based line index of the units row (line 8)
# map: substring to find in the header (case-insensitive)  ->  output name
WANTED = {
    "running time":         "time_s",
    "244.23a force":        "force_kip",
    "244.23a displacement": "displacement_in",
}


def pick_paths():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    in_path = filedialog.askopenfilename(
        title="Select MTS data file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if not in_path:
        root.destroy()
        sys.exit("No file selected.")
    out_dir = filedialog.askdirectory(
        title="Select output folder",
        initialdir=os.path.dirname(in_path))
    root.destroy()
    if not out_dir:
        sys.exit("No output folder selected.")
    return in_path, out_dir


RAMP_KEEP_SECONDS = 5.0   # ramp data kept ahead of a chosen trim-start point
ROLL_WINDOW_SECONDS = 35.0   # width of the scrollable time window
MARKER_MS = 3.5
LINE_W    = 1.0

# same 2-colour slice of the 24-colour palette used in trim.py
_PALETTE = ["#1f77b4", "#ff7f0e"]


def interactive_trim(df, time_col="time_s",
                      force_col="force_kip", disp_col="displacement_in"):
    """Interactive trim window, styled like postprocess_trim.py's marker plot.

    Left-side buttons: Force / Displacement channel toggles, All on / All
    off, Trim Start / Trim End (click a marker button to arm it, click on
    the plot to place — keeps RAMP_KEEP_SECONDS of ramp before the chosen
    start), Skip step (bypasses trimming) and Apply -> (finish). Scroll or
    drag the Time slider to navigate; click a legend entry to toggle that
    channel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    cols_list = [force_col, disp_col]
    time_arr = df[time_col].to_numpy(float)
    data_arr = df[cols_list].to_numpy(float)

    colors = _PALETTE
    vis = [True, True]

    MODE_INFO = {
        "start": ("Trim\nStart", "#007700", "#b0ffb0"),
        "end":   ("Trim\nEnd",   "#cc0000", "#ffb0b0"),
    }
    marks = {k: [None] for k in MODE_INFO}
    mode = [None]
    result = {"skipped": False}
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "Click a marker button to arm it -> click on plot to place  |  "
        "'Skip step' bypasses trimming entirely  •  'Apply ->' when done\n"
        "Scroll to navigate  •  Click legend to toggle channel",
        fontsize=8.5, fontweight="bold")

    plot_ax = fig.add_axes([0.14, 0.13, 0.84, 0.79])
    slide_ax = fig.add_axes([0.14, 0.04, 0.84, 0.04])

    bw, bh, bx = 0.08, 0.05, 0.01
    btn_force = Button(fig.add_axes([bx, 0.85, bw, bh]), "Force",
                       color="#d0e8ff", hovercolor="#b0cfff")
    btn_disp = Button(fig.add_axes([bx, 0.79, bw, bh]), "Displacement",
                      color="#ffe0b0", hovercolor="#ffc870")
    btn_all = Button(fig.add_axes([bx, 0.73, bw, bh]), "All on",
                     color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none = Button(fig.add_axes([bx, 0.67, bw, bh]), "All off",
                      color="#e8e8e8", hovercolor="#d0d0d0")

    mode_axes, mode_btns = {}, {}
    y = 0.58
    for key in ("start", "end"):
        label, _armed_col, idle_col = MODE_INFO[key]
        ax_ = fig.add_axes([bx, y, bw, bh])
        btn_ = Button(ax_, label, color=idle_col, hovercolor=idle_col)
        mode_axes[key] = ax_
        mode_btns[key] = btn_
        y -= 0.06

    btn_skip = Button(fig.add_axes([bx, y - 0.03, bw, bh]), "Skip\nstep",
                      color="#e8e8e8", hovercolor="#d0d0d0")
    btn_apply = Button(fig.add_axes([bx, y - 0.09, bw, bh]), "Apply ->",
                       color="#90ee90", hovercolor="#60dd60")

    for btn in (btn_force, btn_disp, btn_all, btn_none, btn_skip, btn_apply,
                *mode_btns.values()):
        btn.label.set_fontsize(7)

    t_start_full, t_end_full = float(time_arr[0]), float(time_arr[-1])
    window = min(ROLL_WINDOW_SECONDS, t_end_full - t_start_full)
    if window <= 0:
        window = 1.0
    t_max_sl = max(t_end_full - window, t_start_full + 0.01)
    slider = Slider(slide_ax, "Time (s)", t_start_full, t_max_sl,
                    valinit=t_start_full, color="steelblue")

    def update_mode_buttons():
        for key, ax_ in mode_axes.items():
            _label, armed_col, idle_col = MODE_INFO[key]
            armed = (mode[0] == key)
            c = armed_col if armed else idle_col
            mode_btns[key].color = c
            mode_btns[key].hovercolor = c
            mode_btns[key].label.set_color("white" if armed else "black")
            ax_.set_facecolor(c)
        fig.canvas.draw_idle()

    update_mode_buttons()

    def redraw(_=None, keep_xlim=False):
        saved_xlim = plot_ax.get_xlim() if keep_xlim else None
        t_left = slider.val
        t_right = t_left + window
        mask = (time_arr >= t_left) & (time_arr <= t_right)
        win_t = time_arr[mask]

        plot_ax.cla()
        leg_line_map.clear()

        plotted_channels = []
        for i, c in enumerate(cols_list):
            if vis[i]:
                plot_ax.plot(win_t, data_arr[mask, i],
                             lw=LINE_W, color=colors[i], label=c, alpha=0.85,
                             marker="o", markersize=MARKER_MS)
                plotted_channels.append(c)

        for key, (label, col, _idle) in MODE_INFO.items():
            tm = marks[key][0]
            if tm is not None:
                flat_label = label.replace("\n", " ")
                plot_ax.axvline(tm, color=col, lw=2.0, ls="--", zorder=10,
                                label=f"{flat_label}: {tm:.4f} s")

        if mode[0] is not None:
            active_col = MODE_INFO[mode[0]][1]
            flat_label = MODE_INFO[mode[0]][0].replace("\n", " ").upper()
            title_line1 = f"ARMED: {flat_label} — click on plot to place"
        else:
            active_col = "gray"
            title_line1 = "No marker armed — zoom/pan freely"
        plot_ax.set_title(
            f"{title_line1}\nClick a legend entry to toggle that channel",
            fontsize=8.5, color=active_col)

        if keep_xlim and saved_xlim is not None:
            plot_ax.set_xlim(saved_xlim)
        else:
            plot_ax.set_xlim(t_left, t_right)
        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("Value", fontsize=9)
        plot_ax.grid(True, alpha=0.3)

        if plotted_channels or any(v[0] is not None for v in marks.values()):
            leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
            for leg_line, ch_name in zip(leg.get_lines(), plotted_channels):
                leg_line.set_picker(8)
                leg_line.set_linewidth(2.5)
                leg_line_map[leg_line] = ch_name

        fig.canvas.draw_idle()

    def set_group(group):
        if group == "force":
            vis[0], vis[1] = True, False
        elif group == "disp":
            vis[0], vis[1] = False, True
        elif group == "all":
            vis[0], vis[1] = True, True
        elif group == "none":
            vis[0], vis[1] = False, False
        redraw(keep_xlim=True)

    def toggle_mode(m):
        mode[0] = None if mode[0] == m else m
        update_mode_buttons()
        redraw(keep_xlim=True)

    def on_legend_pick(event):
        leg_line = event.artist
        if leg_line not in leg_line_map:
            return
        idx = cols_list.index(leg_line_map[leg_line])
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

    def on_scroll(event):
        if event.inaxes is not plot_ax:
            return
        step = window * 0.1 * (1 if event.button == "down" else -1)
        new_val = min(max(slider.val + step, t_start_full), t_max_sl)
        slider.set_val(new_val)

    def do_skip(_):
        result["skipped"] = True
        plt.close(fig)

    def do_apply(_):
        plt.close(fig)

    slider.on_changed(redraw)
    btn_force.on_clicked(lambda _: set_group("force"))
    btn_disp.on_clicked(lambda _: set_group("disp"))
    btn_all.on_clicked(lambda _: set_group("all"))
    btn_none.on_clicked(lambda _: set_group("none"))
    for key in mode_btns:
        mode_btns[key].on_clicked(lambda _, k=key: toggle_mode(k))
    btn_skip.on_clicked(do_skip)
    btn_apply.on_clicked(do_apply)
    fig.canvas.mpl_connect("pick_event", on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    redraw()
    plt.show()

    if result["skipped"]:
        print("Trim skipped — keeping full dataframe.")
        return df

    start = marks["start"][0]
    end = marks["end"][0]
    if start is None and end is None:
        print("Trim skipped — no markers placed, keeping full dataframe.")
        return df

    start = max(t_start_full, (start - RAMP_KEEP_SECONDS) if start is not None else t_start_full)
    end = end if end is not None else t_end_full

    mask = (df[time_col] >= start) & (df[time_col] <= end)
    trimmed = df.loc[mask].reset_index(drop=True)
    print(f"Trimmed {time_col} to [{start:.3f}, {end:.3f}] s "
          f"(incl. {RAMP_KEEP_SECONDS:.0f}s ramp) -> {len(trimmed)} rows (was {len(df)})")
    return trimmed


def main():
    if len(sys.argv) >= 3:                        # scriptable, no dialogs
        in_path, out_dir = sys.argv[1], sys.argv[2]
    else:
        in_path, out_dir = pick_paths()

    # header on line 7 -> skip the 6 description lines and the units line;
    # utf-8-sig eats the BOM these exports carry
    df = pd.read_csv(
        in_path, sep="\t", encoding="utf-8-sig",
        skiprows=lambda i: i < N_DESC_LINES or i == UNITS_LINE,
        header=0, engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]

    # match wanted channels by case-insensitive substring
    out_cols, missing = {}, []
    for key, out_name in WANTED.items():
        hit = [c for c in df.columns if key in c.lower()]
        if hit:
            out_cols[out_name] = hit[0]
        else:
            missing.append(key)
    if missing:
        sys.exit(f"column(s) not found in header: {missing}\n"
                 f"header seen: {list(df.columns)}")

    out = pd.DataFrame({name: pd.to_numeric(df[src], errors="coerce")
                        for name, src in out_cols.items()})
    n_bad = int(out.isna().any(axis=1).sum())
    if n_bad:
        print(f"NOTE: dropped {n_bad} non-numeric row(s)")
        out = out.dropna().reset_index(drop=True)

    out = interactive_trim(out)

    stem = os.path.splitext(os.path.basename(in_path))[0].strip()
    out_path = os.path.join(out_dir, f"{stem}_extracted.txt")
    out.to_csv(out_path, sep="\t", index=False, float_format="%.6f",
               lineterminator="\r\n")

    t = out["time_s"].to_numpy()
    print(f"\nInput  : {os.path.basename(in_path)}")
    print(f"Rows   : {len(out):,}   time {t[0]:.3f} -> {t[-1]:.3f} s "
          f"({t[-1] - t[0]:.1f} s span)")
    for name in list(WANTED.values())[1:]:
        v = out[name].to_numpy()
        print(f"{name:16s} min {v.min():.4f}   max {v.max():.4f}")
    print(f"Output : {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()