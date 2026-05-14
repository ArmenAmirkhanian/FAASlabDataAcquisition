"""
postprocess_spikes.py — Spike / bad-cycle correction on a trimmed file.

Input : <name>_trimmed.txt  (output of postprocess_trim.py)
Output: <name>_trimmed_clean.txt  +  <name>_trimmed_clean_report.png

The first RAMP_SECONDS of the trimmed file are ramp data and are preserved
unchanged; bad-cycle detection starts from the cycle anchor row onward.

Usage:
    python postprocess_spikes.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_MAD       = 1e-9
MAD_WINDOW_S  = 32
SAMPLE_RATE   = 16
MAD_WINDOW    = MAD_WINDOW_S * SAMPLE_RATE
STATIC_ZSCORE = 8.0
RAMP_SECONDS  = 5.0   # must match postprocess_trim.py

# ── Interactive bad-cycle selector ────────────────────────────────────────────
def select_bad_cycle_markers(time_arr, data_arr, cols_list):
    """
    Shows the trimmed aligned data. User marks the start of the 1st, 2nd, 3rd,
    and 4th bad cycle. Returns (t1, t2, t3, t4) — any can be None if not placed.
    """
    WINDOW_S = 35.0

    MARKERS = [
        ("1st", "#cc6600", "#ffd090", "#ffba60"),
        ("2nd", "#cc0000", "#ff9090", "#ff6060"),
        ("3rd", "#8800cc", "#ddb0ff", "#bb80ff"),
        ("4th", "#0066cc", "#a0c8ff", "#70a8ff"),
    ]
    KEYS = ["1st", "2nd", "3rd", "4th"]

    def ch_color(c):
        if c.startswith("DCDT_"):                               return "steelblue"
        if c.startswith("SG_"):                                 return "tomato"
        if "pressure" in c.lower() or c.startswith("volt_ch"): return "darkorange"
        return "gray"

    colors = [ch_color(c) for c in cols_list]
    vis    = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis): vis[0] = True

    t_marks      = {k: [None] for k in KEYS}
    mode         = [None]
    leg_line_map = {}

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "Arm a marker button → click the start of that bad cycle  |  "
        "Gaps between markers are shown live\n"
        "Place all 4 to see all 3 gaps  •  Close when done",
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

    marker_btn_info = []
    for i, (label, armed_col, idle_col, hover_col) in enumerate(MARKERS):
        y    = 0.51 - i * 0.05
        ax_  = fig.add_axes([bx, y, bw, bh])
        btn_ = Button(ax_, f"{label} bad\ncycle", color=idle_col, hovercolor=hover_col)
        btn_.label.set_fontsize(7)
        marker_btn_info.append((KEYS[i], armed_col, idle_col, btn_, ax_))

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none):
        btn.label.set_fontsize(7)

    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                      valinit=float(time_arr[0]), color="steelblue")

    def update_mode_buttons():
        for key, armed_col, idle_col, btn, ax in marker_btn_info:
            armed = (mode[0] == key)
            c = armed_col if armed else idle_col
            btn.color = c; btn.hovercolor = c
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

        plotted = []
        for i, c in enumerate(cols_list):
            if vis[i]:
                plot_ax.plot(win_t, data_arr[mask, i], lw=0.9, color=colors[i],
                             label=c, alpha=0.85)
                plotted.append(c)

        for key, armed_col, idle_col, btn, ax in marker_btn_info:
            t = t_marks[key][0]
            if t is not None:
                plot_ax.axvline(t, color=armed_col, lw=2, ls="--", zorder=10,
                                label=f"{key}: {t:.3f} s")

        gap_parts  = []
        placed_vals = [t_marks[k][0] for k in KEYS if t_marks[k][0] is not None]
        placed_vals.sort()
        for i in range(len(placed_vals) - 1):
            gap_parts.append(f"gap{i+1}={placed_vals[i+1]-placed_vals[i]:.3f}s")

        if mode[0] is not None:
            col   = next(armed for key, armed, _, _, _ in marker_btn_info if key == mode[0])
            title = f"ARMED: {mode[0].upper()} BAD CYCLE  —  click on plot  |  {'  •  '.join(gap_parts)}"
        else:
            col   = "gray"
            title = f"No marker armed — zoom/pan freely  |  {'  •  '.join(gap_parts)}"

        plot_ax.set_title(f"{title}\nClick legend to toggle channel",
                          fontsize=8.5, color=col)
        if keep_xlim and saved_xlim is not None:
            plot_ax.set_xlim(saved_xlim)
        else:
            plot_ax.set_xlim(t_left, t_right)
        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("Value", fontsize=9)
        plot_ax.grid(True, alpha=0.3)

        if plotted or any(t_marks[k][0] is not None for k in KEYS):
            leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
            for ll, ch_name in zip(leg.get_lines(), plotted):
                ll.set_picker(8); ll.set_linewidth(2.5)
                leg_line_map[ll] = ch_name

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
        ll = event.artist
        if ll not in leg_line_map: return
        idx = cols_list.index(leg_line_map[ll])
        vis[idx] = not vis[idx]
        redraw(keep_xlim=True)

    def on_plot_click(event):
        if mode[0] is None: return
        if event.inaxes is not plot_ax or event.button != 1 or event.xdata is None: return
        leg = plot_ax.get_legend()
        if leg is not None and leg.get_window_extent().contains(event.x, event.y): return
        snapped = float(time_arr[int(np.argmin(np.abs(time_arr - event.xdata)))])
        t_marks[mode[0]][0] = snapped
        redraw(keep_xlim=True)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _: set_group("dcdt"))
    btn_sg.on_clicked(lambda _: set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _: set_group("all"))
    btn_none.on_clicked(lambda _: set_group("none"))
    for key, _, _, btn, _ in marker_btn_info:
        btn.on_clicked(lambda _, k=key: toggle_mode(k))
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    return (t_marks["1st"][0], t_marks["2nd"][0],
            t_marks["3rd"][0], t_marks["4th"][0])


# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

in_path = filedialog.askopenfilename(
    title="Select trimmed file to correct (*_trimmed.txt)",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Trimmed files", "*_trimmed.txt"),
               ("Text files",    "*.txt"),
               ("All files",     "*.*")]
)

if not in_path:
    messagebox.showinfo("Cancelled", "No file selected. Exiting.")
    root.destroy()
    raise SystemExit(0)

root.destroy()

base     = os.path.splitext(in_path)[0]
out_path = base + "_clean.txt"
print(f"Input : {in_path}")
print(f"Output: {out_path}")

# ── Load ──────────────────────────────────────────────────────────────────────
df           = pd.read_csv(in_path, sep="\t")
time_aligned = df["time_s"].values
cols         = [c for c in df.columns if c != "time_s"]
data         = df[cols].values.astype(float)
N, n_ch      = data.shape
dt           = 1.0 / SAMPLE_RATE

cycle_start_idx = int(round(RAMP_SECONDS * SAMPLE_RATE))

print(f"\n{'─'*50}")
print(f"  File            : {os.path.basename(in_path)}")
print(f"  Total rows      : {N:,}")
print(f"  Time range      : {time_aligned[0]:.2f} s → {time_aligned[-1]:.2f} s")
print(f"  Duration        : {time_aligned[-1]:.2f} s  ({time_aligned[-1]/60:.1f} min)")
print(f"  Channels        : {n_ch}")
print(f"  Cycle anchor    : row {cycle_start_idx}  (t = {time_aligned[cycle_start_idx]:.4f} s)")
print(f"{'─'*50}")

# ── Load type ─────────────────────────────────────────────────────────────────
print("\n  Load type:")
print("  [C] Cyclic  — bad-cycle replacement")
print("  [S] Static  — rolling MAD on consecutive differences")

while True:
    load_type = input("\n  Select (C / S): ").strip().upper()
    if load_type in ("C", "S"):
        break
    print("  Enter C or S.")

use_cycle = (load_type == "C")
spc       = None

if use_cycle:
    print("\n  Cyclic loading frequency:")
    print("  [A] Auto-detect via FFT  or  enter Hz (e.g. 0.5, 1.0, 2.0)")

    while True:
        freq_input = input("\n  Frequency (Hz) or A: ").strip().lower()
        if freq_input in ("a", "auto"):
            dcdt_idx  = next((k for k, c in enumerate(cols) if c.startswith("DCDT_")), 0)
            sig       = data[:, dcdt_idx] - data[:, dcdt_idx].mean()
            freqs     = np.fft.rfftfreq(len(sig), d=dt)
            power     = np.abs(np.fft.rfft(sig))
            power[0]  = 0
            peak_freq = freqs[np.argmax(power)]
            print(f"  Auto-detected: {peak_freq:.4f} Hz")
            test_freq = peak_freq
            break
        else:
            try:
                test_freq = float(freq_input)
                if test_freq <= 0:
                    print("  Must be positive.")
                    continue
                break
            except ValueError:
                print("  Enter a number or A.")

    spc           = int(round(1.0 / test_freq / dt))
    n_full_cycles = (N - cycle_start_idx) // spc
    print(f"  Samples per cycle : {spc}  (f = {test_freq:.4f} Hz,  dt = {dt:.4f} s)")
    print(f"  Full cycles from anchor : {n_full_cycles}")

# ── Spike detection ───────────────────────────────────────────────────────────
spike_flags = np.zeros((N, n_ch), dtype=bool)
bad_starts  = []

if use_cycle:
    print("\n  Opening bad-cycle selector.")
    print("  Place up to 4 markers on consecutive bad cycles.")
    print("  The gaps between them are shown live — place all 4 for best accuracy.\n")

    t1, t2, t3, t4 = select_bad_cycle_markers(time_aligned, data, cols)

    placed = [(t, int(np.argmin(np.abs(time_aligned - t))))
              for t in [t1, t2, t3, t4] if t is not None]
    placed.sort(key=lambda x: x[0])

    if len(placed) < 2:
        print("  ERROR: need at least 2 markers placed. Exiting.")
        raise SystemExit(1)

    gaps    = [placed[i+1][1] - placed[i][1] for i in range(len(placed) - 1)]
    avg_gap = int(round(sum(gaps) / len(gaps)))

    print(f"\n  Markers placed: {len(placed)}")
    print(f"  {'Marker':<8} {'Time (s)':>10}   {'Row':>6}")
    for idx, (t, row) in enumerate(placed):
        print(f"  #{idx+1:<7} {t:>10.4f}   {row:>6}")

    print(f"\n  Gaps between consecutive markers:")
    for i, g in enumerate(gaps):
        print(f"    gap {i+1}→{i+2} : {g} rows  ({g / SAMPLE_RATE:.4f} s)")
    print(f"  Average gap   : {avg_gap} rows  ({avg_gap / SAMPLE_RATE:.4f} s)")

    print(f"\n  Period to use for all bad cycles:")
    print(f"  [A] Average gap  ({avg_gap} rows = {avg_gap / SAMPLE_RATE:.4f} s)")
    print(f"  [N] Enter a custom number of rows")
    while True:
        choice = input("\n  Select (A / N): ").strip().upper()
        if choice == "A":
            period_rows = avg_gap
            print(f"  Using average period: {period_rows} rows  ({period_rows / SAMPLE_RATE:.4f} s)")
            break
        elif choice == "N":
            while True:
                try:
                    raw = input(f"  Enter period in rows: ").strip()
                    period_rows = int(raw)
                    if period_rows < 1:
                        print("  Must be >= 1.")
                        continue
                    print(f"  Using custom period: {period_rows} rows  ({period_rows / SAMPLE_RATE:.4f} s)")
                    break
                except ValueError:
                    print("  Enter an integer.")
            break
        else:
            print("  Enter A or N.")

    row_first_bad = placed[0][1]
    r = row_first_bad
    while r + spc <= N:
        bad_starts.append(r)
        r += period_rows

    print(f"\n  Bad cycles identified: {len(bad_starts)}")
    print(f"  {'#':>4}   {'Row':>6}   {'Time (s)':>10}")
    for k, r in enumerate(bad_starts):
        print(f"  {k+1:>4}   {r:>6}   {time_aligned[r]:>10.4f}")

    bad_start_set = set(bad_starts)
    for r in bad_starts:
        spike_flags[r : r + spc, :] = True

    print(f"\n  Flagged {len(bad_starts)} bad cycles for correction.")

else:
    print(f"\nDiff-based spike detection  "
          f"(STATIC_ZSCORE={STATIC_ZSCORE}, MAD window={MAD_WINDOW} samples)...")
    diffs = np.abs(np.diff(data, axis=0))
    rolling_mad = (pd.DataFrame(diffs, columns=cols)
                   .rolling(window=MAD_WINDOW, min_periods=1)
                   .median()
                   .clip(lower=MIN_MAD)
                   .values)
    spike_flags[1:] = diffs > STATIC_ZSCORE * rolling_mad

# ── Spike summary ─────────────────────────────────────────────────────────────
total_spiked   = int(spike_flags.sum())
ch_with_spikes = int((spike_flags.sum(axis=0) > 0).sum())
if use_cycle:
    n_bad_cycles = total_spiked // (spc * n_ch) if (spc * n_ch) > 0 else 0
    print(f"Flagged {n_bad_cycles} bad cycles  "
          f"({n_bad_cycles} × {spc} samples × {n_ch} channels = {total_spiked} channel-samples)")
else:
    print(f"Detected {total_spiked} spiked channel-samples across {ch_with_spikes}/{n_ch} channels")
    for k, c in enumerate(cols):
        n = int(spike_flags[:, k].sum())
        if n > 0:
            print(f"  {c:<30}: {n} spike samples  ({n/N*100:.2f}%)")

if total_spiked == 0:
    print("No spikes found — output will be identical to input.")

# ── Spike correction ──────────────────────────────────────────────────────────
data_clean         = data.copy()
channels_corrected = 0

if use_cycle:
    print(f"\nCorrecting bad cycles: replacing with average of previous + next clean cycle...")
    channels_corrected = n_ch
    bad_start_set      = set(bad_starts)
    for ch in range(n_ch):
        for r in bad_starts:
            if r + spc > N:
                continue
            prev_r = r - period_rows
            while prev_r >= 0 and prev_r in bad_start_set:
                prev_r -= period_rows
            next_r = r + period_rows
            while next_r + spc <= N and next_r in bad_start_set:
                next_r += period_rows
            prev_ok = prev_r >= 0 and prev_r + spc <= N
            next_ok = next_r + spc <= N
            if prev_ok and next_ok:
                data_clean[r:r+spc, ch] = (
                    data[prev_r:prev_r+spc, ch] + data[next_r:next_r+spc, ch]
                ) / 2.0
            elif prev_ok:
                data_clean[r:r+spc, ch] = data[prev_r:prev_r+spc, ch].copy()
            elif next_ok:
                data_clean[r:r+spc, ch] = data[next_r:next_r+spc, ch].copy()

else:
    print("\nCorrecting via linear interpolation  (static load)...")
    for ch in range(n_ch):
        ch_flags = spike_flags[:, ch]
        if not ch_flags.any():
            continue
        channels_corrected += 1
        i = 0
        while i < N:
            if ch_flags[i]:
                run_start = i
                while i < N and ch_flags[i]:
                    i += 1
                run_end  = i
                prev_idx = run_start - 1
                next_idx = run_end
                if prev_idx < 0 and next_idx >= N:
                    pass
                elif prev_idx < 0:
                    data_clean[run_start:run_end, ch] = data_clean[next_idx, ch]
                elif next_idx >= N:
                    data_clean[run_start:run_end, ch] = data_clean[prev_idx, ch]
                else:
                    n_steps = next_idx - prev_idx
                    for j, idx in enumerate(range(run_start, run_end), start=1):
                        alpha = j / n_steps
                        data_clean[idx, ch] = ((1 - alpha) * data_clean[prev_idx, ch]
                                               + alpha     * data_clean[next_idx, ch])
            else:
                i += 1

print(f"Corrected spikes in {channels_corrected}/{n_ch} channels")

# ── Interactive comparison viewer (original vs corrected) ─────────────────────
print("\nOpening comparison plot — review original vs corrected before saving.")
print("Close the window when done reviewing.\n")

COMP_WINDOW_S = 35.0

def _comp_color(c):
    if c.startswith("DCDT_") or c.startswith("volt_ch"): return "steelblue"
    if c.startswith("SG_"):                               return "tomato"
    if "pressure" in c.lower():                           return "darkorange"
    return "gray"

comp_colors = [_comp_color(c) for c in cols]
comp_vis    = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
if not any(comp_vis):
    comp_vis[0] = True

comp_leg_map = {}

fig_comp = plt.figure(figsize=(17, 9))
fig_comp.suptitle(
    f"Review: Original (dashed) vs Corrected (solid)  —  {os.path.basename(in_path)}\n"
    "Close when done  •  Scroll to navigate  •  Group buttons to switch view  "
    "•  Click legend to toggle  •  Red shading = corrected cycles",
    fontsize=8.5, fontweight="bold"
)

cax    = fig_comp.add_axes([0.08, 0.13, 0.90, 0.79])
csax   = fig_comp.add_axes([0.08, 0.04, 0.90, 0.04])

cbw, cbh, cbx = 0.065, 0.048, 0.005
cbtn_dcdt  = Button(fig_comp.add_axes([cbx, 0.80, cbw, cbh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
cbtn_sg    = Button(fig_comp.add_axes([cbx, 0.75, cbw, cbh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
cbtn_press = Button(fig_comp.add_axes([cbx, 0.70, cbw, cbh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
cbtn_all   = Button(fig_comp.add_axes([cbx, 0.65, cbw, cbh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
cbtn_none  = Button(fig_comp.add_axes([cbx, 0.60, cbw, cbh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")
for btn in (cbtn_dcdt, cbtn_sg, cbtn_press, cbtn_all, cbtn_none):
    btn.label.set_fontsize(7)

ct_max  = max(float(time_aligned[-1]) - COMP_WINDOW_S, 0.01)
cslider = Slider(csax, "Time (s)", 0.0, ct_max, valinit=0.0, color="steelblue")

def comp_redraw(_=None, keep_xlim=False):
    saved_xlim = cax.get_xlim() if keep_xlim else None
    t_left  = cslider.val
    t_right = t_left + COMP_WINDOW_S
    mask    = (time_aligned >= t_left) & (time_aligned <= t_right)
    win_t   = time_aligned[mask]

    cax.cla()
    comp_leg_map.clear()

    for i, c in enumerate(cols):
        if not comp_vis[i]:
            continue
        col = comp_colors[i]
        cax.plot(win_t, data[mask, i],       color=col, lw=0.7, ls="--", alpha=0.45)
        cax.plot(win_t, data_clean[mask, i], color=col, lw=1.0, ls="-",  alpha=0.9,
                 label=c, picker=8)

    for r in bad_starts:
        t_bs = time_aligned[r]
        t_be = time_aligned[min(r + spc - 1, N - 1)]
        if t_bs <= t_right and t_be >= t_left:
            cax.axvspan(max(t_bs, t_left), min(t_be, t_right),
                        color="red", alpha=0.10, zorder=0)

    cax.set_xlabel("Time (s)", fontsize=9)
    cax.set_ylabel("Value", fontsize=8)
    cax.set_title("Dashed = original  |  Solid = corrected  |  Red shading = corrected cycles",
                  fontsize=9)
    if keep_xlim and saved_xlim is not None:
        cax.set_xlim(saved_xlim)
    else:
        cax.set_xlim(t_left, t_right)
    cax.grid(True, alpha=0.3)

    plotted = [cols[i] for i in range(len(cols)) if comp_vis[i]]
    if plotted:
        leg = cax.legend(fontsize=7, ncol=2, loc="upper right")
        for ll, ch_name in zip(leg.get_lines(), plotted):
            ll.set_picker(8)
            ll.set_linewidth(2.5)
            comp_leg_map[ll] = ch_name

    fig_comp.canvas.draw_idle()

def comp_set_group(group):
    for i, c in enumerate(cols):
        if   group == "dcdt":  comp_vis[i] = c.startswith("DCDT_") or c.startswith("volt_ch")
        elif group == "sg":    comp_vis[i] = c.startswith("SG_")
        elif group == "press": comp_vis[i] = "pressure" in c.lower()
        elif group == "all":   comp_vis[i] = True
        elif group == "none":  comp_vis[i] = False
    comp_redraw(keep_xlim=True)

def comp_on_legend_pick(event):
    ll = event.artist
    if ll not in comp_leg_map:
        return
    idx = cols.index(comp_leg_map[ll])
    comp_vis[idx] = not comp_vis[idx]
    comp_redraw(keep_xlim=True)

cslider.on_changed(comp_redraw)
cbtn_dcdt.on_clicked(lambda _:  comp_set_group("dcdt"))
cbtn_sg.on_clicked(lambda _:    comp_set_group("sg"))
cbtn_press.on_clicked(lambda _: comp_set_group("press"))
cbtn_all.on_clicked(lambda _:   comp_set_group("all"))
cbtn_none.on_clicked(lambda _:  comp_set_group("none"))
fig_comp.canvas.mpl_connect("pick_event", comp_on_legend_pick)

comp_redraw()
plt.show()

# ── Save prompt ───────────────────────────────────────────────────────────────
print(f"\n  Output files that will be saved:")
print(f"    TXT : {out_path}")
print(f"    PNG : {base + '_clean_report.png'}")

while True:
    save_choice = input("\n  Save output files?  [Y] Yes  /  [N] Discard: ").strip().upper()
    if save_choice in ("Y", "N"):
        break
    print("  Enter Y or N.")

if save_choice == "N":
    print("\n  Discarded — no files saved.")
    raise SystemExit(0)

# ── Write output ──────────────────────────────────────────────────────────────
out_df = pd.DataFrame(data_clean, columns=cols)
out_df.insert(0, "time_s", time_aligned)

with open(out_path, "w") as f:
    f.write("\t".join(out_df.columns) + "\n")
    for row in out_df.itertuples(index=False):
        f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

print(f"\nClean file saved: {out_path}")
print(f"  Rows       : {N:,}")
print(f"  Time range : 0.0000 s → {time_aligned[-1]:.4f} s")

# ── Summary report plot ───────────────────────────────────────────────────────
try:
    combined_flags = spike_flags.any(axis=1)

    fig_sum, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig_sum.suptitle(f"Spike correction — {os.path.basename(in_path)}", fontsize=11)

    ch_idx  = next((k for k, c in enumerate(cols) if c.startswith("DCDT_")), 0)
    ch_name = cols[ch_idx]

    axes[0].plot(time_aligned, data[:, ch_idx],       color="tomato",    lw=0.6, label="Original")
    axes[0].plot(time_aligned, data_clean[:, ch_idx], color="steelblue", lw=0.6, label="Cleaned", alpha=0.8)
    axes[0].axvline(time_aligned[cycle_start_idx], color="darkorange", lw=1.2, ls="--",
                    label=f"Cycle anchor (t = {time_aligned[cycle_start_idx]:.2f} s)")
    for t in time_aligned[spike_flags[:, ch_idx]]:
        axes[0].axvline(t, color="red", lw=0.5, alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{ch_name}  (red = corrected,  orange = cycle anchor)")
    axes[0].legend(fontsize=8)

    axes[1].fill_between(time_aligned, combined_flags.astype(int),
                         color="red", alpha=0.6, step="mid")
    axes[1].set_ylabel("Spike flag\n(any channel)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].set_title("Corrected locations across full test (any channel)")

    plt.tight_layout()
    plot_path = base + "_clean_report.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Report plot saved: {plot_path}")
    plt.show()
except Exception:
    pass
