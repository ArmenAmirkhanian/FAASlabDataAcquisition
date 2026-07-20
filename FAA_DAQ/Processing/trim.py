"""
postprocess_trim.py — Interactive trim and clock-drift alignment.

Opens a raw data file, lets you place:
  • Voltage cycle start  (orange --)  DCDT + Pressure (NI-9205)
  • Strain cycle start   (purple --)  Strain gauges   (NI-9235)
  • File end             (green  | )  optional end trim

Aligns both modules to their cycle starts (keeps RAMP_SECONDS of ramp before
each), corrects clock drift, and writes a single trimmed file with time
starting at 0.

Output: <input_base>_trimmed.txt

Usage:
    python postprocess_trim.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16
RAMP_SECONDS = 5.0   # seconds of ramp data kept before cycle start
MARKER_MS    = 3.5
LINE_W       = 1.0

# ── 24-colour palette (same as process_badcycles_fix.py) ──────────────────────
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]

# ── Optional pre-trim drop-region selector ───────────────────────────────────
def select_drop_markers(time_arr, data_arr, cols_list):
    """
    Optional step, run BEFORE cycle-start selection: mark a bad/garbage
    region to delete outright, separately for the voltage module (DCDT +
    Pressure) and the strain module — e.g. a startup glitch or known-bad
    stretch that would otherwise throw off cycle-start detection.

    Four markers, each independently armed/placed:
      • Voltage drop start / end (orange --)
      • Strain  drop start / end (purple --)
    "Skip step" bypasses this entirely (nothing removed).
    "Apply ->" proceeds with whichever pair(s) were placed; a module with
    no (start, end) pair placed is left untouched.

    Returns (v_drop_start, v_drop_end, s_drop_start, s_drop_end) — any/all
    may be None (None, None, None, None) if skipped or not placed.
    """
    WINDOW_S = 35.0

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cols_list))]
    vis    = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis):
        vis[0] = True

    MODE_INFO = {
        "v_start": ("Voltage\ndrop start", "#cc6600", "#ffd090"),
        "v_end":   ("Voltage\ndrop end",   "#cc6600", "#ffd090"),
        "s_start": ("Strain\ndrop start",  "#7700cc", "#e0b0ff"),
        "s_end":   ("Strain\ndrop end",    "#7700cc", "#e0b0ff"),
    }
    marks        = {k: [None] for k in MODE_INFO}
    mode         = [None]
    result       = {"skipped": False}
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "OPTIONAL — mark a bad/garbage region to delete before trimming, separately for "
        "voltage and strain\n"
        "Click a marker button to arm it -> click on plot to place  |  "
        "'Skip step' bypasses this entirely  •  'Apply ->' when done\n"
        "Scroll to navigate  •  Click legend to toggle channel",
        fontsize=8.5, fontweight="bold"
    )

    plot_ax  = fig.add_axes([0.12, 0.13, 0.85, 0.79])
    slide_ax = fig.add_axes([0.12, 0.04, 0.85, 0.04])

    bw, bh, bx = 0.065, 0.048, 0.005
    btn_dcdt  = Button(fig.add_axes([bx, 0.85, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.80, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.75, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.70, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.65, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

    mode_axes, mode_btns = {}, {}
    y = 0.56
    for key in ("v_start", "v_end", "s_start", "s_end"):
        label, _armed_col, idle_col = MODE_INFO[key]
        ax_  = fig.add_axes([bx, y, bw, bh])
        btn_ = Button(ax_, label, color=idle_col, hovercolor=idle_col)
        mode_axes[key] = ax_
        mode_btns[key] = btn_
        y -= 0.05

    btn_skip  = Button(fig.add_axes([bx, y - 0.03, bw, bh]), "Skip\nstep", color="#e8e8e8", hovercolor="#d0d0d0")
    btn_apply = Button(fig.add_axes([bx, y - 0.08, bw, bh]), "Apply ->",   color="#90ee90", hovercolor="#60dd60")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_skip, btn_apply,
                *mode_btns.values()):
        btn.label.set_fontsize(7)

    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
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

        plotted_channels = []
        for i, c in enumerate(cols_list):
            if vis[i]:
                plot_ax.plot(win_t, data_arr[mask, i],
                             lw=LINE_W, color=colors[i], label=c, alpha=0.85,
                             marker="o", markersize=MARKER_MS)
                plotted_channels.append(c)

        for key, (label, col, _idle) in MODE_INFO.items():
            t = marks[key][0]
            if t is not None:
                flat_label = label.replace("\n", " ")
                plot_ax.axvline(t, color=col, lw=2.0, ls="--", zorder=10,
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
        for i, c in enumerate(cols_list):
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

    def do_skip(_):
        result["skipped"] = True
        plt.close(fig)

    def do_apply(_):
        plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
    btn_sg.on_clicked(lambda _:    set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _:   set_group("all"))
    btn_none.on_clicked(lambda _:  set_group("none"))
    for key in mode_btns:
        mode_btns[key].on_clicked(lambda _, k=key: toggle_mode(k))
    btn_skip.on_clicked(do_skip)
    btn_apply.on_clicked(do_apply)
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    if result["skipped"]:
        return None, None, None, None
    return marks["v_start"][0], marks["v_end"][0], marks["s_start"][0], marks["s_end"][0]


# ── Interactive cycle-start selector ─────────────────────────────────────────
def select_time_markers(time_arr, data_arr, cols_list):
    """
    Three markers placed by clicking:
      • Voltage cycle start (orange --)
      • Strain cycle start  (purple --)
      • File end            (green  | )  — optional
    Returns (t_cs_voltage, t_cs_strain, t_file_end).
    """
    WINDOW_S = 35.0

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cols_list))]
    vis    = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis):
        vis[0] = True

    t_cs_voltage = [None]
    t_cs_strain  = [None]
    t_file_end   = [None]
    mode         = [None]
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "Click a marker button to arm it (lit = active) → click on plot to place  |  "
        "Click armed button again to disarm (zoom/pan freely)  |  File end optional\n"
        "Scroll to navigate  •  Click legend to toggle channel  •  Close when done",
        fontsize=8.5, fontweight="bold"
    )

    plot_ax  = fig.add_axes([0.12, 0.13, 0.85, 0.79])
    slide_ax = fig.add_axes([0.12, 0.04, 0.85, 0.04])

    bw, bh, bx = 0.065, 0.048, 0.005

    btn_dcdt  = Button(fig.add_axes([bx, 0.80, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.75, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.70, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.65, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.60, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

    ax_mcsv = fig.add_axes([bx, 0.51, bw, bh])
    ax_mcss = fig.add_axes([bx, 0.46, bw, bh])
    ax_mfe  = fig.add_axes([bx, 0.41, bw, bh])
    btn_csv = Button(ax_mcsv, "Voltage\nstart", color="#ffd090", hovercolor="#ffba60")
    btn_css = Button(ax_mcss, "Strain\nstart",  color="#e0b0ff", hovercolor="#c880ff")
    btn_fe  = Button(ax_mfe,  "File\nend",      color="#b0ffb0", hovercolor="#80ee80")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none,
                btn_csv, btn_css, btn_fe):
        btn.label.set_fontsize(7)

    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                      valinit=float(time_arr[0]), color="steelblue")

    MODE_ARMED = {
        "cs_voltage": ("#cc6600", "#ffd090"),
        "cs_strain":  ("#7700cc", "#e0b0ff"),
        "file_end":   ("#007700", "#b0ffb0"),
    }

    def update_mode_buttons():
        for m, btn, ax in [
            ("cs_voltage", btn_csv, ax_mcsv),
            ("cs_strain",  btn_css, ax_mcss),
            ("file_end",   btn_fe,  ax_mfe),
        ]:
            armed = (mode[0] == m)
            c = MODE_ARMED[m][0] if armed else MODE_ARMED[m][1]
            btn.color      = c
            btn.hovercolor = c
            btn.label.set_color("white" if armed else "black")
            ax.set_facecolor(c)
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

        if t_cs_voltage[0] is not None:
            plot_ax.axvline(t_cs_voltage[0], color="#cc6600", lw=2.0, ls="--",
                            zorder=10, label=f"Voltage start: {t_cs_voltage[0]:.4f} s")
        if t_cs_strain[0] is not None:
            plot_ax.axvline(t_cs_strain[0], color="#7700cc", lw=2.0, ls="--",
                            zorder=10, label=f"Strain start: {t_cs_strain[0]:.4f} s")
        if t_file_end[0] is not None:
            plot_ax.axvline(t_file_end[0], color="#007700", lw=2.0, ls="-",
                            zorder=10, label=f"File end: {t_file_end[0]:.4f} s")

        parts = []
        if t_cs_voltage[0] is not None: parts.append(f"Voltage start = {t_cs_voltage[0]:.4f} s")
        if t_cs_strain[0]  is not None: parts.append(f"Strain start = {t_cs_strain[0]:.4f} s")
        if t_file_end[0]   is not None: parts.append(f"File end = {t_file_end[0]:.4f} s")

        if mode[0] is not None:
            active_name = {"cs_voltage": "VOLTAGE CYCLE START",
                           "cs_strain":  "STRAIN CYCLE START",
                           "file_end":   "FILE END"}[mode[0]]
            active_col  = MODE_ARMED[mode[0]][0]
            title_line1 = f"ARMED: {active_name}  — click on plot to place  |  {('  •  '.join(parts)) if parts else ''}"
        else:
            active_col  = "gray"
            title_line1 = f"No marker armed — zoom/pan freely  |  {('  •  '.join(parts)) if parts else ''}"
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

        if plotted_channels or any(v[0] is not None for v in [t_cs_voltage, t_cs_strain, t_file_end]):
            leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
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
        if mode[0] == "cs_voltage":
            t_cs_voltage[0] = snapped
        elif mode[0] == "cs_strain":
            t_cs_strain[0] = snapped
        else:
            t_file_end[0] = snapped
        redraw(keep_xlim=True)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _: set_group("dcdt"))
    btn_sg.on_clicked(lambda _: set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _: set_group("all"))
    btn_none.on_clicked(lambda _: set_group("none"))
    btn_csv.on_clicked(lambda _: toggle_mode("cs_voltage"))
    btn_css.on_clicked(lambda _: toggle_mode("cs_strain"))
    btn_fe.on_clicked(lambda _: toggle_mode("file_end"))
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    return t_cs_voltage[0], t_cs_strain[0], t_file_end[0]


# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

in_path = filedialog.askopenfilename(
    title="Select raw data file to trim",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Raw data files", "data_raw_*.txt"),
               ("Text files",     "*.txt"),
               ("All files",      "*.*")]
)

if not in_path:
    messagebox.showinfo("Cancelled", "No file selected. Exiting.")
    root.destroy()
    raise SystemExit(0)

root.destroy()

default_name = os.path.splitext(os.path.basename(in_path))[0] + "_trimmed.txt"
print(f"Input : {in_path}")

# ── Load ──────────────────────────────────────────────────────────────────────
df   = pd.read_csv(in_path, sep="\t")
time = df["time_s"].values

t_start  = time[0]
t_end    = time[-1]
duration = t_end - t_start
n_rows   = len(df)
dt       = np.median(np.diff(time))

print(f"\n{'─'*50}")
print(f"  File         : {os.path.basename(in_path)}")
print(f"  Total rows   : {n_rows:,}")
print(f"  Time range   : {t_start:.2f} s  →  {t_end:.2f} s")
print(f"  Duration     : {duration:.2f} s  ({duration/60:.1f} min)")
print(f"  Sample rate  : {1/dt:.2f} Hz")
print(f"{'─'*50}")

all_cols     = [c for c in df.columns if c != "time_s"]
voltage_cols = [c for c in all_cols if c.startswith("DCDT_")
                                    or "pressure" in c.lower()
                                    or c.startswith("volt_ch")]
strain_cols  = [c for c in all_cols if c.startswith("SG_")]

print(f"\n  Voltage module columns ({len(voltage_cols)}): DCDT + Pressure/Voltage")
print(f"  Strain module columns  ({len(strain_cols)}):  Strain gauges")

# ── Optional: mark & delete a bad/garbage region before trimming ─────────────
# Voltage and strain are deleted independently (own start/end pair each), then
# reconciled to the shorter of the two resulting lengths — same "take the
# shorter" reconciliation the alignment step below already uses. The
# resulting dataframe becomes the input to everything from here on, exactly
# as if it had been the originally loaded file.
print("\n  Opening optional pre-trim 'drop region' plot.")
print("  Mark a bad/garbage stretch to delete — separately for voltage and")
print("  strain if needed — or click 'Skip step' to bypass this entirely.\n")

full_data_for_drop = df[all_cols].values.astype(float)
v_drop_s, v_drop_e, s_drop_s, s_drop_e = select_drop_markers(time, full_data_for_drop, all_cols)
del full_data_for_drop

v_pair_ok = v_drop_s is not None and v_drop_e is not None and v_drop_e > v_drop_s
s_pair_ok = s_drop_s is not None and s_drop_e is not None and s_drop_e > s_drop_s

if not v_pair_ok and not s_pair_ok:
    print("  No drop region applied (skipped, or no valid start/end pair placed).")
else:
    keep_v = np.ones(n_rows, dtype=bool)
    keep_s = np.ones(n_rows, dtype=bool)

    if v_pair_ok:
        row_v_s = int(np.argmin(np.abs(time - v_drop_s)))
        row_v_e = int(np.argmin(np.abs(time - v_drop_e)))
        keep_v[row_v_s:row_v_e] = False
        print(f"  Voltage drop : rows {row_v_s:,} → {row_v_e - 1:,}  "
              f"({row_v_e - row_v_s:,} rows, {(v_drop_e - v_drop_s):.3f} s) removed")
    if s_pair_ok:
        row_s_s = int(np.argmin(np.abs(time - s_drop_s)))
        row_s_e = int(np.argmin(np.abs(time - s_drop_e)))
        keep_s[row_s_s:row_s_e] = False
        print(f"  Strain  drop : rows {row_s_s:,} → {row_s_e - 1:,}  "
              f"({row_s_e - row_s_s:,} rows, {(s_drop_e - s_drop_s):.3f} s) removed")

    v_arr = df[voltage_cols].values[keep_v]
    s_arr = df[strain_cols].values[keep_s]
    n2    = min(len(v_arr), len(s_arr))
    v_arr = v_arr[:n2]
    s_arr = s_arr[:n2]

    new_time2 = np.arange(n2) / SAMPLE_RATE
    out_data2 = {"time_s": new_time2}
    for c in all_cols:
        if c in voltage_cols:
            out_data2[c] = v_arr[:, voltage_cols.index(c)]
        elif c in strain_cols:
            out_data2[c] = s_arr[:, strain_cols.index(c)]

    df   = pd.DataFrame(out_data2)   # becomes the input to the trim step below
    time = df["time_s"].values
    t_start  = time[0]
    t_end    = time[-1]
    duration = t_end - t_start
    n_rows   = len(df)
    dt       = np.median(np.diff(time))
    print(f"  Working data after drop removal: {n_rows:,} rows "
          f"({duration:.2f} s) — this is now the input to the trim step below.")

# ── Interactive marker selection ──────────────────────────────────────────────
print("\n  Opening interactive plot.")
print("  Click 1 → VOLTAGE CYCLE START (orange --)")
print("  Click 2 → STRAIN CYCLE START  (purple --)")
print("  Click 3 → FILE END            (green  | )  — optional")
print("  Close the window when done.\n")

full_data                              = df[all_cols].values.astype(float)
t_cs_voltage, t_cs_strain, t_file_end = select_time_markers(time, full_data, all_cols)
del full_data

print(f"\n  Plot closed.")
if t_cs_voltage is not None: print(f"    Voltage cycle start : {t_cs_voltage:.4f} s")
if t_cs_strain  is not None: print(f"    Strain cycle start  : {t_cs_strain:.4f} s")
if t_file_end   is not None: print(f"    File end            : {t_file_end:.4f} s")

# ── Confirm via terminal ──────────────────────────────────────────────────────
print("\n  Confirm values  (press Enter to accept the value in brackets)\n")

while True:
    try:
        default = t_cs_voltage if t_cs_voltage is not None else t_start
        raw = input(f"  Voltage cycle start (s) [{default:.4f}]: ").strip()
        t_cs_voltage = float(raw) if raw else default
        if t_cs_voltage < t_start or t_cs_voltage > t_end:
            print(f"  Must be between {t_start:.2f} and {t_end:.2f}.")
            continue
        break
    except ValueError:
        print("  Enter a number.")

while True:
    try:
        default = t_cs_strain if t_cs_strain is not None else t_cs_voltage
        raw = input(f"  Strain cycle start  (s) [{default:.4f}]: ").strip()
        t_cs_strain = float(raw) if raw else default
        if t_cs_strain < t_start or t_cs_strain > t_end:
            print(f"  Must be between {t_start:.2f} and {t_end:.2f}.")
            continue
        break
    except ValueError:
        print("  Enter a number.")

# ── File end ──────────────────────────────────────────────────────────────────
print("\n  File end options:")
print("  [1] No trim       — keep all data to end of file")
marker_str = f"{t_file_end:.4f} s" if t_file_end is not None else "not set"
print(f"  [2] Plot marker   — use green marker  ({marker_str})")
print("  [3] Enter time    — type a specific time in seconds")

while True:
    choice = input("\n  Select (1 / 2 / 3): ").strip()
    if choice == "1":
        t_file_end = t_end
        print(f"  Using end of file: {t_file_end:.4f} s")
        break
    elif choice == "2":
        if t_file_end is None:
            print("  No marker was set on the plot. Choose 1 or 3.")
            continue
        if t_file_end <= max(t_cs_voltage, t_cs_strain) or t_file_end > t_end:
            print(f"  Marker at {t_file_end:.4f} s is invalid. Choose 1 or 3.")
            continue
        print(f"  Using plot marker: {t_file_end:.4f} s")
        break
    elif choice == "3":
        while True:
            try:
                raw = input(f"  Enter file end time (s)  [range: {t_start:.2f} – {t_end:.2f}]: ").strip()
                val = float(raw)
                if val <= max(t_cs_voltage, t_cs_strain) or val > t_end:
                    print(f"  Must be > both cycle starts and ≤ {t_end:.2f}.")
                    continue
                t_file_end = val
                print(f"  Using entered time: {t_file_end:.4f} s")
                break
            except ValueError:
                print("  Enter a number.")
        break
    else:
        print("  Enter 1, 2, or 3.")

# ── Compute aligned row ranges ────────────────────────────────────────────────
ramp_rows = int(round(RAMP_SECONDS * SAMPLE_RATE))

row_cs_voltage = int(np.argmin(np.abs(time - t_cs_voltage)))
row_cs_strain  = int(np.argmin(np.abs(time - t_cs_strain)))
row_file_end   = int(np.argmin(np.abs(time - t_file_end)))

volt_start   = max(0, row_cs_voltage - ramp_rows)
strain_start = max(0, row_cs_strain  - ramp_rows)

rows_after_volt   = row_file_end - row_cs_voltage
rows_after_strain = row_file_end - row_cs_strain

n_out_volt   = min(ramp_rows + rows_after_volt,   len(df) - volt_start)
n_out_strain = min(ramp_rows + rows_after_strain, len(df) - strain_start)
n_out        = min(n_out_volt, n_out_strain)

drift_samples = row_cs_strain - row_cs_voltage
drift_ms      = drift_samples * (1.0 / SAMPLE_RATE) * 1000.0

print(f"\n  Clock drift   : {drift_samples:+d} samples  ({drift_ms:+.1f} ms)  "
      f"[strain relative to voltage]")
print(f"  Voltage rows  : {volt_start} → {volt_start + n_out - 1}")
print(f"  Strain rows   : {strain_start} → {strain_start + n_out - 1}")
print(f"  Output rows   : {n_out}  ({n_out / SAMPLE_RATE:.2f} s)")

# ── Build aligned dataset ─────────────────────────────────────────────────────
volt_slice   = df[voltage_cols].values[volt_start   : volt_start   + n_out]
strain_slice = df[strain_cols].values[ strain_start : strain_start + n_out]

new_time = np.arange(n_out) / SAMPLE_RATE

out_data = {"time_s": new_time}
for c in all_cols:
    if c in voltage_cols:
        out_data[c] = volt_slice[:, voltage_cols.index(c)]
    elif c in strain_cols:
        out_data[c] = strain_slice[:, strain_cols.index(c)]

out_df = pd.DataFrame(out_data)

# ── Save-as dialog ───────────────────────────────────────────────────────────
save_root = tk.Tk()
save_root.withdraw()

out_path = filedialog.asksaveasfilename(
    title="Save trimmed output file as...",
    initialdir=os.path.dirname(in_path),
    initialfile=default_name,
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
)

save_root.destroy()

if not out_path:
    print("\n  Save cancelled — no file written.")
    raise SystemExit(0)

# ── Write output ──────────────────────────────────────────────────────────────
with open(out_path, "w") as f:
    f.write("\t".join(out_df.columns) + "\n")
    for row in out_df.itertuples(index=False):
        f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

print(f"\nTrimmed file saved: {out_path}")
print(f"  Rows         : {n_out:,}")
print(f"  Time range   : 0.0000 s → {new_time[-1]:.4f} s")
print(f"  Clock drift  : {drift_ms:+.1f} ms corrected")
print(f"  Ramp kept    : {RAMP_SECONDS:.1f} s  ({ramp_rows} rows) before cycle start")
