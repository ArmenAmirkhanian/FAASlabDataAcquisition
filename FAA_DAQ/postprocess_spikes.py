"""
Spike post-processor — separate cycle-start selection for voltage (DCDT+Pressure)
and strain modules, clock-drift alignment, optional trim, phase-aligned spike
detection (cyclic) or diff-MAD (static), and cycle-phase interpolation correction.
Writes a clean output file with a new time column starting at 0.

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
RAMP_SECONDS         = 5.0   # seconds of ramp data to keep before cycle start

# ── Interactive cycle-start selector ─────────────────────────────────────────
def select_time_markers(time_arr, data_arr, cols_list):
    """
    Three markers are placed by clicking:
      • Voltage cycle start (orange --)  — DCDT + Pressure cycle start (NI-9205)
      • Strain cycle start  (purple --)  — Strain cycle start (NI-9235)
      • File end            (green  | )  — where the output file ends
    Returns (t_cs_voltage, t_cs_strain, t_file_end).
    """
    WINDOW_S = 20.0

    def ch_color(c):
        if c.startswith("DCDT_"):                               return "steelblue"
        if c.startswith("SG_"):                                 return "tomato"
        if "pressure" in c.lower() or c.startswith("volt_ch"): return "darkorange"
        return "gray"

    colors = [ch_color(c) for c in cols_list]
    vis    = [c.startswith("DCDT_") for c in cols_list]
    if not any(vis):
        vis[0] = True

    t_cs_voltage = [None]   # orange dashed
    t_cs_strain  = [None]   # purple dashed
    t_file_end   = [None]   # green solid
    # mode = None means no marker armed; clicks do nothing (zoom/pan work freely)
    # mode = "cs_voltage" / "cs_strain" / "file_end" means that marker is armed
    mode         = [None]
    leg_line_map = {}

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(
        "Click a marker button to arm it (lit = active) → click on plot to place  |  "
        "Click armed button again to disarm (zoom/pan freely)  |  File end optional\n"
        "Scroll to navigate  •  Click legend to toggle channel  •  Close when done",
        fontsize=8.5, fontweight="bold"
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

    # Marker buttons — click to arm (lit), click again to disarm
    ax_mcsv = fig.add_axes([bx, 0.51, bw, bh])
    ax_mcss = fig.add_axes([bx, 0.46, bw, bh])
    ax_mfe  = fig.add_axes([bx, 0.41, bw, bh])
    btn_csv = Button(ax_mcsv, "Voltage\nstart", color="#ffd090", hovercolor="#ffba60")
    btn_css = Button(ax_mcss, "Strain\nstart",  color="#e0b0ff", hovercolor="#c880ff")
    btn_fe  = Button(ax_mfe,  "File\nend",      color="#b0ffb0", hovercolor="#80ee80")

    for btn in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none,
                btn_csv, btn_css, btn_fe):
        btn.label.set_fontsize(7)

    # Slider
    t_max_sl = max(float(time_arr[-1]) - WINDOW_S, float(time_arr[0]) + 0.01)
    slider   = Slider(slide_ax, "Time (s)", float(time_arr[0]), t_max_sl,
                      valinit=float(time_arr[0]), color="steelblue")

    # Armed color (bright) / idle color (pale) for each marker button
    MODE_ARMED = {
        "cs_voltage": ("#cc6600", "#ffd090"),   # (armed, idle)
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

    # ── Draw ────────────────────────────────────────────────────────────────
    def redraw(_=None, keep_xlim=False):
        # Save current x zoom if requested (used when switching channel groups)
        saved_xlim = plot_ax.get_xlim() if keep_xlim else None

        t_left  = slider.val
        t_right = t_left + WINDOW_S
        # Load the full slider window so zooming within it always has data
        mask  = (time_arr >= t_left) & (time_arr <= t_right)
        win_t = time_arr[mask]

        plot_ax.cla()
        leg_line_map.clear()

        plotted_channels = []
        for i, c in enumerate(cols_list):
            if vis[i]:
                plot_ax.plot(win_t, data_arr[mask, i],
                             lw=0.9, color=colors[i], label=c, alpha=0.85)
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
            active_col = MODE_ARMED[mode[0]][0]
            title_line1 = f"ARMED: {active_name}  — click on plot to place  |  {('  •  '.join(parts)) if parts else ''}"
        else:
            active_col  = "gray"
            title_line1 = f"No marker armed — zoom/pan freely  |  {('  •  '.join(parts)) if parts else ''}"
        plot_ax.set_title(
            f"{title_line1}\nClick a legend entry to toggle that channel",
            fontsize=8.5, color=active_col
        )

        # Restore x zoom when switching groups; y always auto-scales to new group
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

    # ── Callbacks ───────────────────────────────────────────────────────────
    def set_group(group):
        for i, c in enumerate(cols_list):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        # keep_xlim=True preserves any toolbar zoom on the time axis when switching groups
        redraw(keep_xlim=True)

    def toggle_mode(m):
        # Click armed button again → disarm; click unarmed button → arm it
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

all_cols = [c for c in df.columns if c != "time_s"]

# Identify which columns belong to which module
voltage_cols = [c for c in all_cols if c.startswith("DCDT_")
                                     or "pressure" in c.lower()
                                     or c.startswith("volt_ch")]
strain_cols  = [c for c in all_cols if c.startswith("SG_")]

print(f"\n  Voltage module columns ({len(voltage_cols)}): DCDT + Pressure/Voltage")
print(f"  Strain module columns  ({len(strain_cols)}):  Strain gauges")

# ── Interactive marker selection ──────────────────────────────────────────────
print("\n  Opening interactive plot.")
print("  Click 1 → VOLTAGE CYCLE START (orange --) — cycle start for DCDT + Pressure")
print("  Click 2 → STRAIN CYCLE START  (purple --) — cycle start for Strain gauges")
print("  Click 3 → FILE END            (green  | ) — optional: set end of file on plot")
print("  File end can also be set after closing (no trim / enter manually).")
print("  Use left-side buttons to switch markers or re-place them.")
print("  Close the window when done.\n")

full_data                          = df[all_cols].values.astype(float)
t_cs_voltage, t_cs_strain, t_file_end = select_time_markers(time, full_data, all_cols)
del full_data

print(f"\n  Plot closed.")
if t_cs_voltage is not None: print(f"    Voltage cycle start : {t_cs_voltage:.4f} s")
if t_cs_strain  is not None: print(f"    Strain cycle start  : {t_cs_strain:.4f} s")
if t_file_end   is not None: print(f"    File end            : {t_file_end:.4f} s")

# ── Confirm via terminal ───────────────────────────────────────────────────────
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

# ── File end: 3 options ───────────────────────────────────────────────────────
print("\n  File end options:")
print("  [1] No trim       — keep all data to end of file")
marker_str = f"{t_file_end:.4f} s" if t_file_end is not None else "not set"
print(f"  [2] Plot marker   — use green marker from plot  ({marker_str})")
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
            print(f"  Marker at {t_file_end:.4f} s is invalid — must be > both cycle starts "
                  f"and ≤ {t_end:.2f}. Choose 1 or 3.")
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

# Each group starts ramp_rows before its own cycle start
volt_start   = max(0, row_cs_voltage - ramp_rows)
strain_start = max(0, row_cs_strain  - ramp_rows)

# Rows from cycle start to file end (file_end is in the original time axis,
# same for both groups — file end marks the same physical moment)
rows_after_volt   = row_file_end - row_cs_voltage
rows_after_strain = row_file_end - row_cs_strain

n_out_volt   = ramp_rows + rows_after_volt
n_out_strain = ramp_rows + rows_after_strain

# Clamp to array bounds
n_out_volt   = min(n_out_volt,   len(df) - volt_start)
n_out_strain = min(n_out_strain, len(df) - strain_start)

# Use the shorter of the two so both groups have the same number of rows
n_out = min(n_out_volt, n_out_strain)

drift_samples = row_cs_strain - row_cs_voltage
drift_ms      = drift_samples * (1.0 / SAMPLE_RATE) * 1000.0

print(f"\n  Clock drift   : {drift_samples:+d} samples  ({drift_ms:+.1f} ms)  "
      f"[strain relative to voltage]")
print(f"  Voltage rows  : {volt_start} → {volt_start + n_out - 1}  "
      f"(cycle start at row {row_cs_voltage})")
print(f"  Strain rows   : {strain_start} → {strain_start + n_out - 1}  "
      f"(cycle start at row {row_cs_strain})")
print(f"  Output rows   : {n_out}  ({n_out / SAMPLE_RATE:.2f} s)")

# ── Build aligned dataset ─────────────────────────────────────────────────────
# Voltage module columns use volt_start offset
# Strain module columns use strain_start offset
# Both windows are the same length (n_out rows)

volt_slice   = df[voltage_cols].values[volt_start   : volt_start   + n_out]
strain_slice = df[strain_cols].values[ strain_start : strain_start + n_out]

# New time column starting at 0
new_time = np.arange(n_out) / SAMPLE_RATE

# Reconstruct in original column order
out_data = {}
out_data["time_s"] = new_time
for c in all_cols:
    if c in voltage_cols:
        idx = voltage_cols.index(c)
        out_data[c] = volt_slice[:, idx]
    elif c in strain_cols:
        idx = strain_cols.index(c)
        out_data[c] = strain_slice[:, idx]

out_df_aligned = pd.DataFrame(out_data)
time_aligned   = out_df_aligned["time_s"].values
cols           = [c for c in out_df_aligned.columns if c != "time_s"]
data           = out_df_aligned[cols].values.astype(float)
N, n_ch        = data.shape

# Cycle start index in the aligned output (ramp_rows rows from the start)
cycle_start_idx = ramp_rows
print(f"\n  Aligned cycle start at output row {cycle_start_idx}  "
      f"(t = {time_aligned[cycle_start_idx]:.4f} s in new time axis)")

# ── Load type ─────────────────────────────────────────────────────────────────
print("\n  Load type:")
print("  [C] Cyclic  — phase-to-phase comparison across 30 cycles")
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

# ── Write output ──────────────────────────────────────────────────────────────
out_df = pd.DataFrame(data_clean, columns=cols)
out_df.insert(0, "time_s", time_aligned)

with open(out_path, "w") as f:
    f.write("\t".join(out_df.columns) + "\n")
    for row in out_df.itertuples(index=False):
        f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

print(f"\nClean file saved: {out_path}")
print(f"  Rows         : {N:,}")
print(f"  Time range   : 0.0000 s → {time_aligned[-1]:.4f} s")
print(f"  Clock drift  : {drift_ms:+.1f} ms corrected")

# ── Summary plot ──────────────────────────────────────────────────────────────
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
    axes[0].set_title(f"{ch_name}  (red = corrected spikes,  orange = cycle anchor)")
    axes[0].legend(fontsize=8)

    axes[1].fill_between(time_aligned, combined_flags.astype(int),
                         color="red", alpha=0.6, step="mid")
    axes[1].set_ylabel("Spike flag\n(any channel)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].set_title("Spike locations across full test (any channel)")

    plt.tight_layout()
    plot_path = base + "_clean_report.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Report plot saved: {plot_path}")
    plt.show()
except Exception:
    pass
