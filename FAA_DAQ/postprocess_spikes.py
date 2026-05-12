"""
Spike post-processor — interactive cycle-start identification, optional trim,
phase-aligned spike detection (cyclic) or diff-MAD (static), and cycle-phase
interpolation correction.  Writes a clean output file.

Usage:
    python postprocess_spikes.py    # opens a file browser to select input file
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── Configuration ─────────────────────────────────────────────────────────────
SPIKE_ZSCORE         = 5.0
PHASE_WINDOW_CYCLES  = 30
MIN_MAD              = 1e-9
MAX_CYCLE_SEARCH     = 10
MAD_WINDOW_S         = 30
SAMPLE_RATE          = 16
MAD_WINDOW           = MAD_WINDOW_S * SAMPLE_RATE
STATIC_ZSCORE        = 8.0
RAMP_SECONDS         = 10.0   # seconds of ramp data to keep before cycle start

# ── Interactive cycle-start selector ─────────────────────────────────────────
def select_time_markers(time_arr, data_arr, cols_list):
    """
    Scrollable interactive plot.  Two markers are placed by clicking:
      • File start  (blue  |  )  — where the output file begins; ramp data is kept
      • Cycle start (orange - -)  — first data point of the first load cycle;
                                    used as the phase anchor for spike detection
    Returns (t_file_start, t_cycle_start).  Either can be None if not clicked.
    """
    WINDOW_S = 20.0

    def ch_color(c):
        if c.startswith("DCDT_"):                               return "steelblue"
        if c.startswith("SG_"):                                 return "tomato"
        if "pressure" in c.lower() or c.startswith("volt_ch"): return "darkorange"
        return "gray"

    colors = [ch_color(c) for c in cols_list]

    vis = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis):
        vis[0] = True

    t_cycle_start = [None]          # orange dashed line
    t_file_end    = [None]          # green solid line
    mode          = ["cycle_start"] # which marker the next click places
    leg_line_map  = {}

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "Click 1 = Cycle start (orange --)  •  Click 2 = File end (green |)  |  "
        "Scroll to navigate  •  Click legend to toggle channel  •  Close when done",
        fontsize=9, fontweight="bold"
    )

    plot_ax  = fig.add_axes([0.08, 0.13, 0.90, 0.79])
    slide_ax = fig.add_axes([0.08, 0.04, 0.90, 0.04])

    bw, bh, bx = 0.065, 0.048, 0.005

    # Channel group buttons
    btn_dcdt  = Button(fig.add_axes([bx, 0.80, bw, bh]), "DCDT",     color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.75, bw, bh]), "Strain",   color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.70, bw, bh]), "Pressure", color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.65, bw, bh]), "All on",   color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.60, bw, bh]), "All off",  color="#e8e8e8", hovercolor="#d0d0d0")

    # Marker mode buttons
    ax_mcs = fig.add_axes([bx, 0.51, bw, bh])
    ax_mfe = fig.add_axes([bx, 0.46, bw, bh])
    btn_mode_cs = Button(ax_mcs, "Set\nCycle start", color="#ffd090", hovercolor="#ffba60")
    btn_mode_fe = Button(ax_mfe, "Set\nFile end",    color="#b0ffb0", hovercolor="#80ee80")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none,
                btn_mode_cs, btn_mode_fe):
        btn.label.set_fontsize(7)

    # Slider
    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                      valinit=float(time_arr[0]), color="steelblue")

    def update_mode_buttons():
        btn_mode_cs.color      = "#cc8010" if mode[0] == "cycle_start" else "#ffd090"
        btn_mode_cs.hovercolor = "#bb6f00" if mode[0] == "cycle_start" else "#ffba60"
        btn_mode_fe.color      = "#009900" if mode[0] == "file_end"    else "#b0ffb0"
        btn_mode_fe.hovercolor = "#007700" if mode[0] == "file_end"    else "#80ee80"
        btn_mode_cs.label.set_color("white" if mode[0] == "cycle_start" else "black")
        btn_mode_fe.label.set_color("white" if mode[0] == "file_end"    else "black")
        ax_mcs.set_facecolor(btn_mode_cs.color)
        ax_mfe.set_facecolor(btn_mode_fe.color)

    update_mode_buttons()

    # ── Draw ────────────────────────────────────────────────────────────────
    def redraw(_=None):
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
                             lw=0.9, color=colors[i], label=c, alpha=0.85)
                plotted_channels.append(c)

        # Cycle-start marker — dashed orange vertical line
        if t_cycle_start[0] is not None:
            plot_ax.axvline(t_cycle_start[0], color="#cc6600", lw=2.0, ls="--",
                            zorder=10, label=f"Cycle start: {t_cycle_start[0]:.4f} s")

        # File-end marker — solid green vertical line
        if t_file_end[0] is not None:
            plot_ax.axvline(t_file_end[0], color="#007700", lw=2.0, ls="-",
                            zorder=10, label=f"File end: {t_file_end[0]:.4f} s")

        # Status line
        parts = []
        if t_cycle_start[0] is not None: parts.append(f"Cycle start = {t_cycle_start[0]:.4f} s  (orange --)")
        if t_file_end[0]    is not None: parts.append(f"File end = {t_file_end[0]:.4f} s  (green |)")
        active_name = {"cycle_start": "CYCLE START", "file_end": "FILE END"}[mode[0]]
        active_col  = {"cycle_start": "#cc6600",     "file_end": "#007700"}[mode[0]]
        status = "  •  ".join(parts) if parts else ""
        plot_ax.set_title(
            f"Next click places:  {active_name}  —  {status}\n"
            "Click a legend entry to toggle that channel",
            fontsize=8.5, color=active_col
        )

        plot_ax.set_xlim(t_left, t_right)
        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("Value", fontsize=9)
        plot_ax.grid(True, alpha=0.3)

        if plotted_channels or t_cycle_start[0] is not None or t_file_end[0] is not None:
            leg = plot_ax.legend(fontsize=7, ncol=2, loc="upper right")
            for leg_line, ch_name in zip(leg.get_lines(), plotted_channels):
                leg_line.set_picker(8)
                leg_line.set_linewidth(2.5)
                leg_line_map[leg_line] = ch_name

        fig.canvas.draw_idle()

    # ── Callbacks ───────────────────────────────────────────────────────────
    def set_group(group):
        for i, c in enumerate(cols_list):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw()

    def set_mode(m):
        mode[0] = m
        update_mode_buttons()
        fig.canvas.draw_idle()

    def on_legend_pick(event):
        leg_line = event.artist
        if leg_line not in leg_line_map:
            return
        idx = cols_list.index(leg_line_map[leg_line])
        vis[idx] = not vis[idx]
        redraw()

    def on_plot_click(event):
        if event.inaxes is not plot_ax or event.button != 1 or event.xdata is None:
            return
        leg = plot_ax.get_legend()
        if leg is not None and leg.get_window_extent().contains(event.x, event.y):
            return
        snapped = float(time_arr[int(np.argmin(np.abs(time_arr - event.xdata)))])
        if mode[0] == "cycle_start":
            t_cycle_start[0] = snapped
            mode[0] = "file_end"      # auto-advance after 1st click
        else:
            t_file_end[0] = snapped
        update_mode_buttons()
        redraw()

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _: set_group("dcdt"))
    btn_sg.on_clicked(lambda _: set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _: set_group("all"))
    btn_none.on_clicked(lambda _: set_group("none"))
    btn_mode_cs.on_clicked(lambda _: set_mode("cycle_start"))
    btn_mode_fe.on_clicked(lambda _: set_mode("file_end"))
    fig.canvas.mpl_connect("pick_event",         on_legend_pick)
    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    redraw()
    plt.show()

    return t_cycle_start[0], t_file_end[0]


# ── File browser ──────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.withdraw()

in_path = filedialog.askopenfilename(
    title="Select raw data file to process",
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

base     = os.path.splitext(in_path)[0]
out_path = base + "_clean.txt"
print(f"Input : {in_path}")
print(f"Output: {out_path}")

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

all_cols  = [c for c in df.columns if c != "time_s"]

# ── Interactive marker selection (full un-trimmed data) ───────────────────────
print("\n  Opening interactive plot.")
print("  First click  → CYCLE START (orange --) — first data point of first load cycle")
print("  Second click → FILE END    (green |)   — where output file ends")
print("  Use left-side buttons to switch marker or re-click to move either line.")
print("  Close the window when done.")

full_data                    = df[all_cols].values.astype(float)
t_cycle_start, t_file_end   = select_time_markers(time, full_data, all_cols)
del full_data

print(f"\n  Plot closed.")
if t_cycle_start is not None: print(f"    Cycle start : {t_cycle_start:.4f} s")
if t_file_end    is not None: print(f"    File end    : {t_file_end:.4f} s")

# ── Confirm / trim ────────────────────────────────────────────────────────────
print("\n  Confirm values  (press Enter to accept the value in brackets)\n")

while True:
    try:
        default = t_cycle_start if t_cycle_start is not None else t_start
        raw = input(f"  Cycle start (s) [{default:.4f}]: ").strip()
        t_cycle_start = float(raw) if raw else default
        if t_cycle_start < t_start or t_cycle_start > t_end:
            print(f"  Must be between {t_start:.2f} and {t_end:.2f}.")
            continue
        break
    except ValueError:
        print("  Enter a number.")

while True:
    try:
        default_end = t_file_end if t_file_end is not None else t_end
        raw = input(f"  File end    (s) [{default_end:.4f}]: ").strip()
        t_file_end = float(raw) if raw else default_end
        if t_file_end <= t_cycle_start or t_file_end > t_end:
            print(f"  Must be greater than cycle start ({t_cycle_start:.2f}) and ≤ {t_end:.2f}.")
            continue
        break
    except ValueError:
        print("  Enter a number.")

t_file_start = max(t_start, t_cycle_start - RAMP_SECONDS)
actual_ramp  = t_cycle_start - t_file_start

print(f"\n  File start  : {t_file_start:.4f} s  "
      f"({actual_ramp:.2f} s of ramp before cycle start,  "
      f"RAMP_SECONDS={RAMP_SECONDS})")
print(f"  Cycle start : {t_cycle_start:.4f} s")
print(f"  File end    : {t_file_end:.4f} s")

mask = (time >= t_file_start) & (time <= t_file_end)
df   = df[mask].reset_index(drop=True)
time = df["time_s"].values
print(f"\n  Kept {len(df):,} rows  "
      f"({time[0]:.4f} s → {time[-1]:.4f} s,  "
      f"{time[-1]-time[0]:.2f} s total duration)")

cols    = [c for c in df.columns if c != "time_s"]
data    = df[cols].values.astype(float)
N, n_ch = data.shape
dt      = np.median(np.diff(time))

print(f"\nProcessing {N:,} samples, {n_ch} channels")

# ── Load type ─────────────────────────────────────────────────────────────────
print("\n  Load type:")
print("  [C] Cyclic  — phase-to-phase comparison across 30 cycles")
print("  [S] Static  — rolling MAD on consecutive differences")

while True:
    load_type = input("\n  Select (C / S): ").strip().upper()
    if load_type in ("C", "S"):
        break
    print("  Enter C or S.")

use_cycle       = (load_type == "C")
spc             = None
cycle_start_idx = 0

if use_cycle:
    # ── Frequency ─────────────────────────────────────────────────────────────
    print("\n  Cyclic loading frequency:")
    print("  [A] Auto-detect via FFT  or  enter Hz (e.g. 0.5, 1.0, 2.0)")

    while True:
        freq_input = input("\n  Frequency (Hz) or A: ").strip().lower()
        if freq_input in ("a", "auto"):
            dcdt_idx = next((k for k, c in enumerate(cols) if c.startswith("DCDT_")), 0)
            sig      = data[:, dcdt_idx] - data[:, dcdt_idx].mean()
            freqs    = np.fft.rfftfreq(len(sig), d=dt)
            power    = np.abs(np.fft.rfft(sig))
            power[0] = 0
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
    n_full_cycles = N // spc
    print(f"  Samples per cycle : {spc}  (f = {test_freq:.4f} Hz,  dt = {dt:.4f} s)")
    print(f"  Total full cycles : {n_full_cycles}")

    # Resolve cycle start index relative to the (possibly trimmed) time array
    if t_cycle_start is not None:
        cycle_start_idx = int(np.searchsorted(time, t_cycle_start))
        # Clamp to valid range
        cycle_start_idx = max(0, min(cycle_start_idx, N - 1))
        print(f"  Cycle anchor      : t = {time[cycle_start_idx]:.4f} s  "
              f"(sample {cycle_start_idx})")
        print(f"  Full cycles from anchor : {(N - cycle_start_idx) // spc}")
    else:
        cycle_start_idx = 0
        print(f"  Cycle anchor      : t = {time[0]:.4f} s  (no anchor selected — using start)")

    if n_full_cycles < PHASE_WINDOW_CYCLES:
        print(f"  WARNING: fewer than {PHASE_WINDOW_CYCLES} full cycles — "
              f"reference window reduced automatically.")

# ── Spike detection ───────────────────────────────────────────────────────────
spike_flags = np.zeros((N, n_ch), dtype=bool)

if use_cycle:
    print(f"\nPhase-aligned spike detection  "
          f"(SPIKE_ZSCORE={SPIKE_ZSCORE}, window={PHASE_WINDOW_CYCLES} cycles)...")
    min_per = max(3, PHASE_WINDOW_CYCLES // 6)

    for ch in range(n_ch):
        for p in range(spc):
            first = cycle_start_idx + p
            if first >= N:
                continue
            indices = np.arange(first, N, spc)
            vals    = pd.Series(data[indices, ch])

            ref_med   = vals.rolling(PHASE_WINDOW_CYCLES, center=True,
                                     min_periods=min_per).median()
            deviation = (vals - ref_med).abs()
            roll_mad  = deviation.rolling(PHASE_WINDOW_CYCLES, center=True,
                                          min_periods=min_per).median()
            roll_mad  = roll_mad.clip(lower=MIN_MAD)

            spike_flags[indices, ch] = (deviation > SPIKE_ZSCORE * roll_mad).values

    print("Phase-aligned detection complete.")

else:
    print(f"\nDiff-based spike detection  "
          f"(SPIKE_ZSCORE={STATIC_ZSCORE}, MAD window={MAD_WINDOW} samples)...")
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
print(f"Detected {total_spiked} spiked channel-samples across {ch_with_spikes}/{n_ch} channels")
for k, c in enumerate(cols):
    n = int(spike_flags[:, k].sum())
    if n > 0:
        print(f"  {c:<30}: {n} spike samples  ({n/N*100:.2f}%)")

if total_spiked == 0:
    print("No spikes found — output file will be identical to input.")

# ── Spike correction ──────────────────────────────────────────────────────────
data_clean         = data.copy()
channels_corrected = 0

if use_cycle:
    print(f"\nCorrecting via cycle-phase interpolation  (spc={spc})...")
    for ch in range(n_ch):
        ch_spike_idx = np.where(spike_flags[:, ch])[0]
        if len(ch_spike_idx) == 0:
            continue
        channels_corrected += 1
        for i in ch_spike_idx:
            prev_val = None
            for k in range(1, MAX_CYCLE_SEARCH + 1):
                pi = i - k * spc
                if pi < 0:
                    break
                if not spike_flags[pi, ch]:
                    prev_val = data_clean[pi, ch]
                    break
            next_val = None
            for k in range(1, MAX_CYCLE_SEARCH + 1):
                ni = i + k * spc
                if ni >= N:
                    break
                if not spike_flags[ni, ch]:
                    next_val = data_clean[ni, ch]
                    break
            if prev_val is not None and next_val is not None:
                data_clean[i, ch] = (prev_val + next_val) / 2.0
            elif prev_val is not None:
                data_clean[i, ch] = prev_val
            elif next_val is not None:
                data_clean[i, ch] = next_val

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
                run_end = i
                prev_idx, next_idx = run_start - 1, run_end
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

# ── Write output ──────────────────────────────────────────────────────────────
out_df = pd.DataFrame(data_clean, columns=cols)
out_df.insert(0, "time_s", time)

with open(out_path, "w") as f:
    f.write("\t".join(out_df.columns) + "\n")
    for row in out_df.itertuples(index=False):
        f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

print(f"Clean file saved: {out_path}")

# ── Summary plot ──────────────────────────────────────────────────────────────
try:
    combined_flags = spike_flags.any(axis=1)

    fig_sum, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig_sum.suptitle(f"Spike correction — {os.path.basename(in_path)}", fontsize=11)

    ch_idx  = next((k for k, c in enumerate(cols) if c.startswith("DCDT_")), 0)
    ch_name = cols[ch_idx]

    axes[0].plot(time, data[:, ch_idx],       color="tomato",    lw=0.6, label="Original")
    axes[0].plot(time, data_clean[:, ch_idx], color="steelblue", lw=0.6, label="Cleaned", alpha=0.8)
    if use_cycle and cycle_start_idx > 0:
        axes[0].axvline(time[cycle_start_idx], color="green", lw=1.2, ls="--",
                        label=f"Cycle anchor ({time[cycle_start_idx]:.2f} s)")
    for t in time[spike_flags[:, ch_idx]]:
        axes[0].axvline(t, color="red", lw=0.5, alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{ch_name}  (red = corrected spikes,  green = cycle anchor)")
    axes[0].legend(fontsize=8)

    axes[1].fill_between(time, combined_flags.astype(int), color="red", alpha=0.6, step="mid")
    axes[1].set_ylabel("Spike flag\n(any channel)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].set_title("Spike locations across full test (any channel)")

    plt.tight_layout()
    plot_path = base + "_clean_report.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Report plot saved: {plot_path}")
    plt.show()
except ImportError:
    pass
