"""
postprocess_peaks.py — Phase-alignment via zero-crossing detection.

WHY ZERO-CROSSINGS (not peaks)
  A flat-topped signal has peak position uncertainty of ±2-3 rows, so
  peak-to-peak gaps naturally range from 10-22 rows — making threshold
  detection unreliable.  The zero-crossing (upward pass through the signal
  midpoint) occurs on a steep slope, so its row position is precise to
  < 0.5 rows.  A dropout of N samples gives gap = SPC - N (e.g. 9 rows
  for N=7) versus a clean gap of 16.000 ± 0.5 — clearly detectable.

WORKFLOW
  1. File browser  — select trimmed input file
  2. Auto-detect   — find upward zero-crossings in each DCDT channel,
                     flag gaps < SPC - GAP_TOLERANCE
  3. Merge & print — cluster same-event detections into single confirmed events
  4. Insert rows   — linear interpolation inserts exactly N_missing rows
  5. Review plot   — INPUT (solid) + OUTPUT (dashed), green lines = insertions
  6. Save to folder

CONFIGURATION  (top of file)
  GAP_TOLERANCE  — rows below SPC to flag (SPC - GAP_TOLERANCE = 13 default)
  MIN_CH_CONFIRM — DCDT channels that must agree (1 = any channel)
  MERGE_WINDOW   — rows within which detections are merged into one event

Usage:
    python postprocess_peaks.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16      # Hz
CYCLE_HZ       = 1.0    # Hz
RAMP_SECONDS   = 5.0    # seconds to skip at start (not cyclic)
GAP_TOLERANCE  = 3      # flag gaps < SPC - GAP_TOLERANCE  (i.e. < 13 for SPC=16)
MIN_CH_CONFIRM = 1      # DCDT channels that must agree on an event
MERGE_WINDOW   = 6      # rows: detections within this window → single event
                        # DCDT inter-channel spread for same event: ≤ 4-5 rows
                        # consecutive bad-cycle separation:         ≥ 7 rows (SPC-max_miss)
                        # → 6 groups same-event channels while separating consecutive

WINDOW_S  = 20.0        # seconds visible in review plot
MARKER_MS = 4
LINE_W    = 0.9

# ── Derived ───────────────────────────────────────────────────────────────────
SPC       = int(round(SAMPLE_RATE / CYCLE_HZ))   # rows per cycle = 16
RAMP_ROWS = int(RAMP_SECONDS * SAMPLE_RATE)       # rows to skip  = 80

# ── 24-colour palette ─────────────────────────────────────────────────────────
_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39",
]

# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
in_path = filedialog.askopenfilename(
    title="Select trimmed input file",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Trimmed files", "*_trimmed.txt"),
               ("Text files",    "*.txt"),
               ("All files",     "*.*")]
)
if not in_path:
    messagebox.showinfo("Cancelled", "No file selected.")
    root.destroy()
    raise SystemExit(0)
root.destroy()

# ── Load ──────────────────────────────────────────────────────────────────────
df      = pd.read_csv(in_path, sep="\t")
cols    = [c for c in df.columns if c != "time_s"]
data    = df[cols].values.astype(float)
N, n_ch = data.shape
time_in = np.arange(0, N) / SAMPLE_RATE

det_idx = [i for i, c in enumerate(cols) if c.startswith("DCDT_")]
if not det_idx:
    det_idx = list(range(min(4, n_ch)))   # fallback if no DCDT_ columns found

chan_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_ch)]

print(f"\nInput      : {os.path.basename(in_path)}")
print(f"Rows       : {N:,}  ({time_in[-1]:.2f} s)")
print(f"Channels   : {n_ch}  |  detection on: {[cols[i] for i in det_idx]}")
print(f"SPC        : {SPC} rows/cycle  |  flagging gaps < {SPC - GAP_TOLERANCE} rows")


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-CROSSING FINDER  — upward crossings of the signal midpoint
#   Uses linear interpolation for sub-sample precision.
#   Crossings occur on the steepest slope → position uncertainty < 0.5 rows.
# ══════════════════════════════════════════════════════════════════════════════
def find_upcrossings(signal, level):
    """
    Return fractional row indices where signal crosses 'level' going upward.
    Linear interpolation gives sub-sample precision.
    """
    out = []
    for i in range(len(signal) - 1):
        if signal[i] < level <= signal[i + 1]:
            frac = (level - signal[i]) / (signal[i + 1] - signal[i])
            out.append(i + frac)
    return np.array(out)


# ══════════════════════════════════════════════════════════════════════════════
# GAP DETECTION  — short crossing-to-crossing intervals signal a dropout
#   Stores the exact start/end row of the compressed cycle so we can
#   replace it entirely with the previous clean cycle.
# ══════════════════════════════════════════════════════════════════════════════
GAP_MAX    = SPC - GAP_TOLERANCE     # flag gaps shorter than this (e.g. < 13)
raw_events = []    # (cycle_start, cycle_end, n_missing, channel_name)

for ch_i in det_idx:
    sig = data[:, ch_i]

    # Crossing level = midpoint of oscillation in the cyclic portion
    cyc   = sig[RAMP_ROWS:]
    level = (float(np.percentile(cyc, 5)) + float(np.percentile(cyc, 95))) / 2.0

    # Find all upward crossings in the cyclic portion, shift back to full indices
    xings = find_upcrossings(cyc, level) + RAMP_ROWS

    for j in range(1, len(xings)):
        gap = xings[j] - xings[j - 1]
        if 0 < gap < GAP_MAX:
            n_miss      = int(round(SPC - gap))
            cycle_start = int(round(xings[j - 1]))   # start of compressed cycle
            cycle_end   = int(round(xings[j]))        # end   of compressed cycle
            if n_miss > 0:
                raw_events.append((cycle_start, cycle_end, n_miss, cols[ch_i]))

print(f"\nRaw detections : {len(raw_events)}"
      f"  ({len(det_idx)} DCDT channels, upward zero-crossings)")


# ══════════════════════════════════════════════════════════════════════════════
# MERGE  — cluster nearby detections from different channels into one event
# ══════════════════════════════════════════════════════════════════════════════
raw_events.sort(key=lambda x: x[0])

all_groups    = []    # every group before MIN_CH_CONFIRM filter
merged_events = []    # (cycle_start, cycle_end, n_missing, [channels], n_votes)
i = 0
while i < len(raw_events):
    group = [raw_events[i]]
    j     = i + 1
    while j < len(raw_events) and raw_events[j][0] - raw_events[i][0] <= MERGE_WINDOW:
        group.append(raw_events[j])
        j += 1

    cycle_start = int(np.median([e[0] for e in group]))
    cycle_end   = int(np.median([e[1] for e in group]))
    n_miss      = int(round(np.median([e[2] for e in group])))
    ch_names    = list({e[3] for e in group})
    n_votes     = len(ch_names)

    all_groups.append(n_votes)
    if n_votes >= MIN_CH_CONFIRM and n_miss > 0:
        merged_events.append((cycle_start, cycle_end, n_miss, ch_names, n_votes))
    i = j

# Diagnostic: vote distribution
vote_counts = {}
for v in all_groups:
    vote_counts[v] = vote_counts.get(v, 0) + 1
print(f"\nGroup vote distribution (before MIN_CH_CONFIRM={MIN_CH_CONFIRM} filter):")
for v in sorted(vote_counts):
    flag = "  ← accepted" if v >= MIN_CH_CONFIRM else "  ← rejected"
    print(f"  {v} channel(s) agreed : {vote_counts[v]:>6} groups{flag}")

print(f"\nMerged events  : {len(merged_events)}  "
      f"(confirmed by {MIN_CH_CONFIRM}+ DCDT channels out of {len(det_idx)} total)")
print()
print(f"  {'#':>4}  {'Cycle start':>11}  {'Time (s)':>10}  "
      f"{'N recover':>10}  {'Votes':>6}  Channels")
print(f"  {'─'*4}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*28}")
for k, (cs, ce, nm, chs, nv) in enumerate(merged_events):
    print(f"  {k+1:>4}  {cs:>11,}  {cs / SAMPLE_RATE:>10.3f}  "
          f"{nm:>10}  {nv:>6}  {', '.join(chs)}")

if not merged_events:
    print("\nNo dropout events detected — output would be identical to input.")
    raise SystemExit(0)


# ══════════════════════════════════════════════════════════════════════════════
# REPLACE COMPRESSED CYCLES  — adaptive fix for 1 or 2 consecutive bad cycles
#
#   First, consecutive bad-cycle pairs are merged into compound events so
#   their replacements do not overlap.  Then each compound event is fixed
#   by erasing N bad cycles + 1 good cycle and pasting N+1 clean cycles.
#
#   Single bad cycle  (N = 1):
#     Erase  : bad cycle + 1 good after  →  2·SPC − n_miss  rows removed
#     Insert : 2 previous clean cycles   →  2·SPC            rows added
#     Net    : +n_miss
#
#   Two consecutive bad cycles  (N = 2):
#     Erase  : 2 bad + 1 good after      →  3·SPC − (n1+n2)  rows removed
#     Insert : 3 previous clean cycles   →  3·SPC             rows added
#     Net    : +(n1 + n2)
#
#   Processing in DESCENDING row order keeps all earlier indices valid.
# ══════════════════════════════════════════════════════════════════════════════
insert_times_in = [cs / SAMPLE_RATE for cs, *_ in merged_events]

# Set of known bad cycle-start rows for quick lookup
bad_starts = {cs for cs, *_ in merged_events}

# ── Merge consecutive bad-cycle pairs into compound events ────────────────────
sorted_asc     = sorted(merged_events, key=lambda x: x[0])
compound_events = []   # each entry: (cycle_start, replace_end, n_miss_total, n_ref)

i = 0
while i < len(sorted_asc):
    cs, ce, nm, _, _ = sorted_asc[i]
    # Two consecutive bad cycles: next event starts within MERGE_WINDOW of ce
    if (i + 1 < len(sorted_asc)
            and abs(sorted_asc[i + 1][0] - ce) <= MERGE_WINDOW):
        cs2, ce2, nm2, _, _ = sorted_asc[i + 1]
        compound_events.append((cs, ce2 + SPC, nm + nm2, 3))  # erase 2 bad + 1 good
        i += 2
    else:
        compound_events.append((cs, ce + SPC, nm, 2))          # erase 1 bad + 1 good
        i += 1

n_single = sum(1 for ev in compound_events if ev[3] == 2)
n_double = sum(1 for ev in compound_events if ev[3] == 3)
print(f"\nFix plan   : {len(compound_events)} repairs  "
      f"— {n_single} single bad cycle,  {n_double} double consecutive")

# ── Apply fixes in descending row order ───────────────────────────────────────
output_data = data.copy()
sorted_desc = sorted(compound_events, key=lambda x: x[0], reverse=True)

for cycle_start, replace_end, n_miss_total, n_ref in sorted_desc:
    if replace_end > len(output_data) or cycle_start < RAMP_ROWS:
        continue

    # Walk back until n_ref consecutive clean cycles are all outside bad_starts
    ref_start = cycle_start - n_ref * SPC
    while (any((ref_start + k * SPC) in bad_starts for k in range(n_ref))
           and ref_start >= RAMP_ROWS):
        ref_start -= SPC
    if ref_start < RAMP_ROWS:
        continue

    prev_ref = output_data[ref_start : ref_start + n_ref * SPC].copy()

    output_data = np.vstack([
        output_data[:cycle_start],
        prev_ref,                   # n_ref · SPC rows
        output_data[replace_end:]   # skip bad cycle(s) + trailing SPC rows
    ])

M              = len(output_data)
new_time       = np.arange(0, M) / SAMPLE_RATE
total_inserted = M - N

print(f"\n{'─'*54}")
print(f"  Detected   : {len(merged_events)} bad cycles  "
      f"→  {len(compound_events)} repairs  ({n_single} single, {n_double} double)")
print(f"  Inserted   : +{total_inserted} rows  ({total_inserted / SAMPLE_RATE:.4f} s recovered)")
print(f"  Input rows : {N:,}  ({time_in[-1]:.4f} s)")
print(f"  Output rows: {M:,}  ({new_time[-1]:.4f} s)")
print(f"{'─'*54}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT REVIEW PLOT
#   INPUT  = solid line + dots    OUTPUT = dashed line + dots   (same colour)
#   Green dashed verticals = insertion locations
#   Buttons: channel groups, Next/Prev 20 s, Save, Discard
#   Click legend line to toggle individual channel
# ══════════════════════════════════════════════════════════════════════════════
def show_output_review():
    state        = {'decision': None}
    leg_line_map = {}
    vis = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
    if not any(vis):
        vis = [True] + [False] * (len(cols) - 1)

    fig      = plt.figure(figsize=(17, 9))
    plot_ax  = fig.add_axes([0.08, 0.13, 0.88, 0.75])
    slide_ax = fig.add_axes([0.08, 0.04, 0.88, 0.035])

    bw, bh, bx = 0.055, 0.042, 0.004
    btn_dcdt    = Button(fig.add_axes([bx, 0.83, bw, bh]), "DCDT",
                         color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg      = Button(fig.add_axes([bx, 0.78, bw, bh]), "Strain",
                         color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press   = Button(fig.add_axes([bx, 0.73, bw, bh]), "Pressure",
                         color="#ffe0b0", hovercolor="#ffc870")
    btn_all     = Button(fig.add_axes([bx, 0.68, bw, bh]), "All on",
                         color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none    = Button(fig.add_axes([bx, 0.63, bw, bh]), "All off",
                         color="#e8e8e8", hovercolor="#d0d0d0")
    btn_next    = Button(fig.add_axes([bx, 0.55, bw, bh]), "Next\n20s",
                         color="#dde8ff", hovercolor="#bbd0ff")
    btn_prev    = Button(fig.add_axes([bx, 0.50, bw, bh]), "Prev\n20s",
                         color="#dde8ff", hovercolor="#bbd0ff")
    btn_save    = Button(fig.add_axes([bx, 0.44, bw, bh]), "Save",
                         color="#90ee90", hovercolor="#60dd60")
    btn_discard = Button(fig.add_axes([bx, 0.39, bw, bh]), "Discard",
                         color="#ffaaaa", hovercolor="#ff8888")
    for b in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none,
              btn_next, btn_prev, btn_save, btn_discard):
        b.label.set_fontsize(7)

    t_data_max = max(float(time_in[-1]), float(new_time[-1]))
    t_max      = max(t_data_max - WINDOW_S, 0.01)
    slider     = Slider(slide_ax, "Time (s)", 0.0, t_max,
                        valinit=0.0, color="steelblue")

    fig.suptitle(
        f"OUTPUT REVIEW — {os.path.basename(in_path)}  |  "
        f"{len(merged_events)} insertions  (+{total_inserted} rows)  |  "
        f"Input {N:,} → Output {M:,} rows\n"
        "——  INPUT (original)     - - -  OUTPUT (fixed)     "
        "same colour = same channel     green dashed = insertion point     "
        "click legend to toggle channel",
        fontsize=8.5, fontweight="bold"
    )

    def redraw(_=None):
        t0       = slider.val
        t1       = t0 + WINDOW_S
        mask_in  = (time_in  >= t0) & (time_in  <= t1)
        mask_out = (new_time >= t0) & (new_time <= t1)

        plot_ax.cla()
        y_labels = []
        vis_idx  = [i for i in range(n_ch) if vis[i]]

        for i in vis_idx:
            c   = cols[i]
            clr = chan_colors[i]
            plot_ax.plot(time_in[mask_in], data[mask_in, i],
                         color=clr, lw=LINE_W, linestyle="-",
                         marker="o", markersize=MARKER_MS,
                         alpha=0.90, label=f"{c} (in)")
            plot_ax.plot(new_time[mask_out], output_data[mask_out, i],
                         color=clr, lw=LINE_W, linestyle="--",
                         marker="o", markersize=MARKER_MS,
                         alpha=0.65, label=f"{c} (out)")
            if c.startswith("DCDT_") and "Displacement" not in y_labels:
                y_labels.append("Displacement")
            elif ("pressure" in c.lower() or c.startswith("volt_ch")) \
                    and "Pressure/Voltage" not in y_labels:
                y_labels.append("Pressure/Voltage")
            elif c.startswith("SG_") and "Strain" not in y_labels:
                y_labels.append("Strain")

        for t_ins in insert_times_in:
            if t0 - 1 <= t_ins <= t1 + 1:
                plot_ax.axvline(t_ins, color="green", lw=1.2,
                                ls="--", alpha=0.65, zorder=3)

        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("  /  ".join(y_labels) if y_labels else "Value",
                           fontsize=8)
        plot_ax.set_xlim(t0, t1)
        plot_ax.grid(True, alpha=0.25)

        leg_line_map.clear()
        if vis_idx:
            leg       = plot_ax.legend(fontsize=6, ncol=3, loc="upper right")
            leg_lines = leg.get_lines()
            for j, chan_i in enumerate(vis_idx):
                for k in range(2):          # 0=(in), 1=(out)
                    ll_idx = j * 2 + k
                    if ll_idx < len(leg_lines):
                        leg_lines[ll_idx].set_picker(8)
                        leg_lines[ll_idx].set_linewidth(2.5)
                        leg_line_map[leg_lines[ll_idx]] = chan_i

        fig.canvas.draw_idle()

    def set_group(group):
        for i, c in enumerate(cols):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = ("pressure" in c.lower()
                                             or c.startswith("volt_ch"))
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw()

    def on_legend_pick(event):
        ll = event.artist
        if ll not in leg_line_map:
            return
        vis[leg_line_map[ll]] = not vis[leg_line_map[ll]]
        redraw()

    def do_next(_):    slider.set_val(min(slider.val + 20.0, t_max))
    def do_prev(_):    slider.set_val(max(slider.val - 20.0, 0.0))
    def on_save(_):    state['decision'] = True;  plt.close(fig)
    def on_discard(_): state['decision'] = False; plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:    set_group("dcdt"))
    btn_sg.on_clicked(lambda _:      set_group("sg"))
    btn_press.on_clicked(lambda _:   set_group("press"))
    btn_all.on_clicked(lambda _:     set_group("all"))
    btn_none.on_clicked(lambda _:    set_group("none"))
    btn_next.on_clicked(do_next)
    btn_prev.on_clicked(do_prev)
    btn_save.on_clicked(on_save)
    btn_discard.on_clicked(on_discard)
    fig.canvas.mpl_connect("pick_event", on_legend_pick)

    redraw()
    plt.show()
    return state['decision']


print("\nOpening output review plot...")
save_it = show_output_review()

if not save_it:
    print("\nDiscarded — no file written.")
    raise SystemExit(0)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE — folder picker, filename auto-generated
# ══════════════════════════════════════════════════════════════════════════════
default_name = os.path.splitext(os.path.basename(in_path))[0] + "_aligned.txt"
save_root = tk.Tk()
save_root.withdraw()
save_dir = filedialog.askdirectory(
    title="Select folder to save aligned file",
    initialdir=os.path.dirname(in_path)
)
save_root.destroy()
if not save_dir:
    print("\nSave cancelled — no file written.")
    raise SystemExit(0)
out_path = os.path.join(save_dir, default_name)

print(f"\nWriting: {out_path}")
with open(out_path, "w") as f:
    f.write("time_s\t" + "\t".join(cols) + "\n")
    for i in range(M):
        row_vals = ([f"{new_time[i]:.6f}"]
                    + [f"{v:.6f}" for v in output_data[i]])
        f.write("\t".join(row_vals) + "\n")

print(f"Done — {M:,} rows written.")
print(f"  Time : {new_time[0]:.4f} s → {new_time[-1]:.4f} s")
