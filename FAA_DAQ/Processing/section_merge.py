"""
section_merge.py — Pick up to 2 sections from each of several input files
(DCDT/voltage and Strain marked independently, since they can drift out of
phase), then merge all picked sections end-to-end into one file.

For each file you select (any order, add as many as you like):
  1. An interactive plot opens — same slider/legend/group-toggle interface as
     postprocess_trim.py / postprocess_merge.py.
  2. Arm one of the 8 markers (Section 1/2 x DCDT/Strain x Start/End) and
     click on the plot to place it. DCDT and Strain are independent within
     each section, so the two modules can be trimmed to their own cycle
     boundaries. You don't have to use both sections or both groups — place
     only what you need and click Apply.

After every file has been processed, the DCDT chunks and Strain chunks are
each concatenated separately, in the order picked (file order, then Section
1 -> Section 2 within each file) — raw values, no shifting or scaling. The
two groups are then combined into one output file with a continuous
time_s; if they end up different lengths, the shorter one is padded with
blank rows at the end.

Usage:
    python section_merge.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox
from _dialog_utils import get_last_dir, set_last_dir

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16    # Hz — used to rebuild a continuous time_s after merging
MARKER_MS    = 3.5
LINE_W       = 1.0
WINDOW_S     = 15.0

# ── 24-colour palette (same as other Processing scripts) ─────────────────────
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]

# (section, group) -> (button label, armed colour, idle colour)
MARK_INFO = {
    (1, "dcdt"):   ("Sec1 DCDT",   "#cc6600", "#ffd090"),
    (1, "strain"): ("Sec1 Strain", "#7700cc", "#e0b0ff"),
    (2, "dcdt"):   ("Sec2 DCDT",   "#0066cc", "#a8d0ff"),
    (2, "strain"): ("Sec2 Strain", "#cc0066", "#ffb0d8"),
}


def is_voltage_col(c):
    return c.startswith("DCDT_") or "pressure" in c.lower() or c.startswith("volt_ch")


def is_strain_col(c):
    return c.startswith("SG_")


# ── Interactive per-file section selector ────────────────────────────────────
def select_sections(time_arr, data_arr, cols_list, fname):
    """
    Up to 2 sections, each with independent DCDT and Strain start/end pairs
    (8 markers total, each independently armed/placed):
      Section 1 DCDT start/end   (orange)
      Section 1 Strain start/end (purple)
      Section 2 DCDT start/end   (blue)
      Section 2 Strain start/end (magenta)
    Start = dashed line, End = dotted line.
    "Apply ->" proceeds with whichever complete pairs were placed; groups
    left empty (or with only one marker) are simply not used.

    Returns a dict {(section, group): (start_t, end_t)} for whichever pairs
    got a valid start < end.
    """
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cols_list))]
    vis    = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis):
        vis[0] = True

    marks = {key: {"start": [None], "end": [None]} for key in MARK_INFO}
    mode  = [None]   # e.g. ((1, "dcdt"), "start")
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        f"{fname}\n"
        "Click a marker button to arm it -> click on plot to place  |  "
        "Click armed button again to disarm  |  DCDT and Strain are independent per section\n"
        "Scroll to navigate  •  Click legend to toggle channel  •  Click 'Apply ->' when done",
        fontsize=8.5, fontweight="bold"
    )

    plot_ax  = fig.add_axes([0.14, 0.13, 0.84, 0.79])
    slide_ax = fig.add_axes([0.14, 0.04, 0.84, 0.04])

    bw, bh, bx = 0.075, 0.045, 0.005
    btn_dcdt  = Button(fig.add_axes([bx, 0.85, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.80, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.75, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.70, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.65, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

    mode_axes, mode_btns = {}, {}
    y = 0.56
    for sec in (1, 2):
        for group in ("dcdt", "strain"):
            label, _armed_col, idle_col = MARK_INFO[(sec, group)]
            for part in ("start", "end"):
                ax_  = fig.add_axes([bx, y, bw, bh])
                btn_ = Button(ax_, f"{label}\n{part}", color=idle_col, hovercolor=idle_col)
                mode_axes[((sec, group), part)] = ax_
                mode_btns[((sec, group), part)] = btn_
                y -= 0.045
        y -= 0.015

    btn_apply = Button(fig.add_axes([bx, y - 0.02, bw, bh]), "Apply ->", color="#90ee90", hovercolor="#60dd60")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_apply, *mode_btns.values()):
        btn.label.set_fontsize(6.5)

    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                      valinit=float(time_arr[0]), color="steelblue")

    def update_mode_buttons():
        for (key, part), ax_ in mode_axes.items():
            _label, armed_col, idle_col = MARK_INFO[key]
            armed = (mode[0] == (key, part))
            c = armed_col if armed else idle_col
            mode_btns[(key, part)].color      = c
            mode_btns[(key, part)].hovercolor = c
            mode_btns[(key, part)].label.set_color("white" if armed else "black")
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

        plotted_channels = []
        for i, c in enumerate(cols_list):
            if vis[i]:
                plot_ax.plot(win_t, data_arr[mask, i],
                             lw=LINE_W, color=colors[i], label=c, alpha=0.85,
                             marker="o", markersize=MARKER_MS)
                plotted_channels.append(c)

        any_marker = False
        for key, (label, col, _idle) in MARK_INFO.items():
            t0 = marks[key]["start"][0]
            t1 = marks[key]["end"][0]
            if t0 is not None:
                any_marker = True
                plot_ax.axvline(t0, color=col, lw=2.0, ls="--", zorder=10,
                                label=f"{label} start: {t0:.4f} s")
            if t1 is not None:
                any_marker = True
                plot_ax.axvline(t1, color=col, lw=2.0, ls=":", zorder=10,
                                label=f"{label} end: {t1:.4f} s")

        if mode[0] is not None:
            key, part = mode[0]
            active_col  = MARK_INFO[key][1]
            title_line1 = f"ARMED: {MARK_INFO[key][0].upper()} {part.upper()} — click on plot to place"
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
        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("Value", fontsize=9)
        plot_ax.grid(True, alpha=0.3)

        if plotted_channels or any_marker:
            leg = plot_ax.legend(fontsize=6.5, ncol=2, loc="upper right")
            for leg_line, ch_name in zip(leg.get_lines(), plotted_channels):
                leg_line.set_picker(8)
                leg_line.set_linewidth(2.5)
                leg_line_map[leg_line] = ch_name

        fig.canvas.draw_idle()

    def set_group(group):
        for i, c in enumerate(cols_list):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw(keep_xlim=True)

    def toggle_mode(k):
        mode[0] = None if mode[0] == k else k
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
        key, part = mode[0]
        marks[key][part][0] = snapped
        redraw(keep_xlim=True)

    def do_apply(_):
        plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
    btn_sg.on_clicked(lambda _:    set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _:   set_group("all"))
    btn_none.on_clicked(lambda _:  set_group("none"))
    for k in mode_btns:
        mode_btns[k].on_clicked(lambda _, kk=k: toggle_mode(kk))
    btn_apply.on_clicked(do_apply)
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    result = {}
    for key in MARK_INFO:
        t0 = marks[key]["start"][0]
        t1 = marks[key]["end"][0]
        if t0 is not None and t1 is not None and t1 > t0:
            result[key] = (t0, t1)
    return result


def pad_to(series, n):
    arr = series.to_numpy(dtype=float)
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def main():
    root = tk.Tk()
    root.withdraw()

    in_paths = []
    while True:
        path = filedialog.askopenfilename(
            title=f"Select data file #{len(in_paths) + 1}  (Cancel to stop adding files)",
            initialdir=os.path.dirname(in_paths[-1]) if in_paths else get_last_dir(),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            break
        in_paths.append(path)
        set_last_dir(path)
        if not messagebox.askyesno(
            "Add another file?",
            f"{len(in_paths)} file(s) selected so far:\n"
            + "\n".join(os.path.basename(p) for p in in_paths)
            + "\n\nAdd another file?"
        ):
            break

    if not in_paths:
        messagebox.showinfo("Cancelled", "No files selected. Exiting.")
        root.destroy()
        return

    print(f"\nSelected {len(in_paths)} file(s), in this order:")
    for p in in_paths:
        print(f"  {os.path.basename(p)}")

    v_chunks, s_chunks = [], []
    voltage_cols = strain_cols = None

    for p in in_paths:
        df = pd.read_csv(p, sep="\t")
        data_cols = [c for c in df.columns if c != "time_s"]

        if voltage_cols is None:
            voltage_cols = [c for c in data_cols if is_voltage_col(c)]
            strain_cols  = [c for c in data_cols if is_strain_col(c)]
            other_cols   = [c for c in data_cols if c not in voltage_cols and c not in strain_cols]
            if other_cols:
                print(f"  Note: unclassified columns {other_cols} will be treated as DCDT/voltage group.")
                voltage_cols = voltage_cols + other_cols

        time = df["time_s"].to_numpy(float) if "time_s" in df.columns else np.arange(len(df), dtype=float)

        fname = os.path.basename(p)
        print(f"\n{fname}: {len(df)} rows — opening section picker...")
        full_data = df[data_cols].to_numpy(float)
        picks = select_sections(time, full_data, data_cols, fname)
        del full_data

        if not picks:
            print(f"  No sections placed for {fname} — file skipped.")
            continue

        for sec in (1, 2):
            for group, cols, chunks_list in (("dcdt", voltage_cols, v_chunks), ("strain", strain_cols, s_chunks)):
                key = (sec, group)
                if key not in picks or not cols:
                    continue
                t0, t1 = picks[key]
                row0 = int(np.argmin(np.abs(time - t0)))
                row1 = int(np.argmin(np.abs(time - t1)))
                chunk = df[cols].iloc[row0:row1 + 1].reset_index(drop=True)
                print(f"  Section {sec} {group}: rows {row0}-{row1}  ({len(chunk)} rows, {t1 - t0:.3f} s)")
                chunks_list.append(chunk)

    if not v_chunks and not s_chunks:
        print("\nNo sections were placed in any file. Nothing to merge. Exiting.")
        root.destroy()
        return

    v_merged = pd.concat(v_chunks, ignore_index=True) if v_chunks else pd.DataFrame(columns=voltage_cols)
    s_merged = pd.concat(s_chunks, ignore_index=True) if s_chunks else pd.DataFrame(columns=strain_cols)
    n_v, n_s = len(v_merged), len(s_merged)
    n_total = max(n_v, n_s)
    if n_v != n_s:
        print(f"\nNote: DCDT group has {n_v} rows, Strain group has {n_s} rows after merging "
              f"— the shorter group is padded with blank rows at the end.")

    merged = pd.DataFrame()
    merged["time_s"] = np.arange(n_total) / SAMPLE_RATE
    for c in voltage_cols:
        merged[c] = pad_to(v_merged[c], n_total) if c in v_merged.columns else np.full(n_total, np.nan)
    for c in strain_cols:
        merged[c] = pad_to(s_merged[c], n_total) if c in s_merged.columns else np.full(n_total, np.nan)

    out_path = filedialog.asksaveasfilename(
        title="Save merged output file as...",
        initialdir=os.path.dirname(in_paths[0]),
        initialfile="section_merged.txt",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    root.destroy()

    if not out_path:
        print("\n  Save cancelled — no file written.")
        return

    set_last_dir(out_path)

    with open(out_path, "w") as f:
        f.write("\t".join(merged.columns) + "\n")
        for row in merged.itertuples(index=False):
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

    print(f"\nDone — {len(merged)} rows written to:\n  {out_path}")
    print(f"  Time range: 0.0000 s -> {merged['time_s'].iloc[-1]:.4f} s")


if __name__ == "__main__":
    main()
