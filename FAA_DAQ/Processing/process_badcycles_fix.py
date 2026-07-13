#!/usr/bin/env python3
"""
fix_bad_cycles.py — Dropped-sample detection and repair for STRATA fatigue
DAQ files (16 Hz sampling, 1 Hz half-sine loading -> 16 samples per cycle),
with an interactive color-coded review viewer.

REQUIREMENTS:  numpy, pandas, matplotlib  (tkinter ships with Python).
               No scipy needed.

WHAT THE BAD CYCLES ARE
  Every ~9.7 s the DAQ drops a chunk of samples mid-cycle. The surviving
  points are genuine but the waveform skips ahead in phase, so the cycle
  looks compressed (7-9 points) and non-sinusoidal. The two DAQ modules
  lose DIFFERENT amounts: group 1 (DCDT_* + volt_* channels) loses k
  samples per event (7-12, median 8); group 2 (SG_* channels) loses k+2
  (9-14, median 10) and its jump lands 4-5 rows earlier.

DETECTION (per channel group, on a reference channel)
  In clean data every sample equals the sample 16 rows earlier (same phase,
  one cycle later). A dropped-sample event breaks that for exactly 16 rows.
  The first row of the mismatch burst is the jump row. The number of
  dropped samples k is found by sliding the last clean cycle (template)
  forward 1..15 samples and keeping the best fit. Load start/stop
  transients are rejected by requiring the pre-event template to be a
  full-amplitude cycle.

REPAIR (mode "insert", default)
  Each module's columns are rebuilt as an independent stream: at every
  event the k lost samples are re-inserted. Each inserted value is the
  AVERAGE of the real samples at the same cycle phase one cycle before
  and one cycle after. This restores complete 16-point / 1.000 s cycles
  and the true test clock. Group 2 gains ~2 more rows per event than
  group 1; both streams are merged from t=0 and trimmed to the shorter,
  then time_s is rebuilt at exact 1/16 s spacing.

USAGE
  python fix_bad_cycles.py                      -> file browser + review plot
  python fix_bad_cycles.py input.txt            -> review plot, save via dialog
  python fix_bad_cycles.py input.txt -o out.txt -> saves to given path
  Options:
    --clean-output PATH   also write a copy without the inserted_* flag columns
    --eventlog PATH       write the event table (group, jump_row, k) as CSV
    --mode insert|replace|drop   (insert is the validated default)
    --no-plot             headless: skip the review viewer, save directly
    --no-flags            omit inserted_*/repaired_* columns from the output
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 16      # Hz
CYCLE_HZ    = 1.0     # Hz
P           = int(round(SAMPLE_RATE / CYCLE_HZ))   # samples per cycle = 16
TH_FRAC     = 0.15    # periodicity-error threshold, fraction of amplitude
GATE_FRAC   = 0.5     # pre-event template must span this fraction of amplitude
WINDOW_S    = 20.0    # seconds visible in the review plot
MARKER_MS   = 3.5
LINE_W      = 1.0

# Channel groups: one DAQ module each; detection runs on the reference channel
GROUPS = {
    "DCDT": dict(prefix="DCDT_", ref="DCDT_Beam_B2_Top"),
    "VOLT": dict(prefix="volt_", ref="volt_ch18"),
    "SG":   dict(prefix="SG_",   ref="SG_4E_top"),
}
# Streams for insert mode: DCDT+volt share a module; SG is the other module
STREAMS = {
    "G1": ("DCDT_", "volt_"),
    "G2": ("SG_",),
}
STREAM_REF = {"G1": "DCDT", "G2": "SG"}   # which event list drives each stream

# ── 24-colour palette (same as postprocess_peaks.py) ──────────────────────────
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION  (logic unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def detect_events(x, start=2 * P, th_frac=TH_FRAC):
    """Find dropped-sample events on a reference channel.

    Periodicity error e[i] = |x[i] - x[i-16]| is ~0 during steady cycling
    and bursts for exactly 16 rows after a dropped-sample jump.
    Returns list of (jump_row, k) with k = number of samples skipped.
    """
    n = len(x)
    amp = np.percentile(x, 99) - np.percentile(x, 1)
    err = np.zeros(n)
    err[P:] = np.abs(x[P:] - x[:-P])
    bad = err > th_frac * amp

    events, i = [], start
    while i < n:
        if bad[i]:
            j = i
            if j < P:
                i += 1
                continue
            T = x[j - P:j]                       # last clean cycle = template
            if (T.max() - T.min()) < GATE_FRAC * amp:
                i = j + P                        # load start/stop, not a drop
                continue
            W = min(P, n - j)
            best_k, best_e = None, np.inf
            for k in range(1, P):
                pred = T[(np.arange(W) + k) % P]
                e = np.mean((x[j:j + W] - pred) ** 2)
                if e < best_e:
                    best_e, best_k = e, k
            events.append((j, best_k))
            i = j + P
        else:
            i += 1
    return events


# ══════════════════════════════════════════════════════════════════════════════
# REPAIR — insert mode (logic unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def repair_insert(df, group_events):
    """Rebuild each module's stream by inserting the k lost samples at every
    event. Inserted values = average of the samples at the same cycle phase
    one cycle before and one cycle after (both real data). Streams merged
    from t=0, trimmed to the shorter, time_s rebuilt at exact 1/16 s."""
    n = len(df)
    streams = {}
    for g, prefixes in STREAMS.items():
        streams[g] = dict(
            cols=[c for c in df.columns if c.startswith(prefixes)],
            events=group_events[STREAM_REF[g]],
        )

    rebuilt, flags = {}, {}
    for g, spec in streams.items():
        parts = {c: [] for c in spec["cols"]}
        fparts, prev = [], 0
        for (j, k) in spec["events"]:
            m = np.arange(k)
            for c in spec["cols"]:
                x = df[c].to_numpy()
                parts[c].append(x[prev:j])
                parts[c].append(0.5 * (x[j - P + m] + x[j + P - k + m]))
            fparts.append(np.zeros(j - prev, np.int8))
            fparts.append(np.ones(k, np.int8))
            prev = j
        for c in spec["cols"]:
            parts[c].append(df[c].to_numpy()[prev:])
        fparts.append(np.zeros(n - prev, np.int8))
        rebuilt[g] = {c: np.concatenate(parts[c]) for c in spec["cols"]}
        flags[g] = np.concatenate(fparts)

    L = min(len(flags[g]) for g in streams)
    out = pd.DataFrame({"time_s": np.arange(L) / float(SAMPLE_RATE)})
    for c in df.columns:
        if c == "time_s":
            continue
        g = next(g for g, pfx in STREAMS.items() if c.startswith(pfx))
        out[c] = rebuilt[g][c][:L]
    for g in streams:
        out[f"inserted_{g}"] = flags[g][:L]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# REPAIR — legacy modes (replace / drop), logic unchanged.
# "replace" needs scipy; it is imported lazily so insert/drop run without it.
# ══════════════════════════════════════════════════════════════════════════════
def periodic_template(vals):
    """Cubic periodic spline through one 16-sample cycle (replace mode only)."""
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        sys.exit("mode 'replace' needs scipy (pip install scipy); "
                 "the default 'insert' mode does not.")
    p = np.arange(P + 1, dtype=float)
    v = np.append(vals, vals[0])
    return CubicSpline(p, v, bc_type="periodic")


def repair_replace(df, group_events, group_cols):
    """In-place rewrite of the 16 affected rows per event (compressed-cycle
    bridge). Kept for reference; superseded by insert mode."""
    flags = {g: np.zeros(len(df), dtype=np.int8) for g in group_events}
    for g, events in group_events.items():
        cols = group_cols[g]
        for (j, k) in events:
            if j < P or j + P > len(df):
                continue
            rows = np.arange(j, j + P)
            m = np.arange(P)
            phase = (15.0 + (m + 1) * (17.0 + k) / 17.0) % P
            for c in cols:
                x = df[c].to_numpy()
                f = periodic_template(x[j - P:j])
                bridge = f(phase)
                if j + P < len(df):
                    eps = x[j + P] - float(f((16.0 + k) % P))
                    bridge = bridge + (m + 1) / 17.0 * eps
                df.loc[rows, c] = bridge
            flags[g][rows] = 1
    return flags


def repair_drop(df, group_events):
    """Delete rows per event, phase-matched to the DCDT/volt group.
    NOTE: leaves a 2-sample phase kink in SG channels at each splice."""
    ev_d = {j: k for j, k in group_events["DCDT"]}
    ev_s = dict(group_events["SG"])
    drop_mask = np.zeros(len(df), dtype=bool)
    log = []
    sg_rows = np.array(sorted(ev_s.keys())) if ev_s else np.array([])
    for j_d, k_d in sorted(ev_d.items()):
        j_start = j_d
        if len(sg_rows):
            near = sg_rows[(sg_rows >= j_d - 8) & (sg_rows <= j_d + 8)]
            if len(near):
                j_start = min(j_d, int(near[0]))
        L = 32 - k_d
        drop_mask[j_start:j_start + L] = True
        log.append((j_start, L, k_d))
    out = df.loc[~drop_mask].reset_index(drop=True)
    out["time_s"] = np.arange(len(out)) / float(SAMPLE_RATE)
    return out, log


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate(df, label):
    """Continuity check: worst sample-to-sample step / amplitude per group.
    Clean cyclic data stays below ~0.45 even for steep channels."""
    report = {}
    for g, spec in GROUPS.items():
        c = spec["ref"]
        if c not in df.columns:
            continue
        x = df[c].to_numpy()
        amp = np.percentile(x, 99) - np.percentile(x, 1)
        report[g] = round(float(np.abs(np.diff(x)).max() / amp), 3)
    print(f"[{label}] worst step / amplitude per group (<=~0.45 is clean): {report}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE REVIEW VIEWER  (style of postprocess_peaks.py)
#   INPUT = solid + dots      OUTPUT = dashed + dots     same colour per channel
#   Green dashed verticals = repair (event) locations, input-time coordinates
#   Buttons: DCDT / Strain / Pressure / All on / All off / Next / Prev
#   Time slider, clickable legend, Save / Discard
# ══════════════════════════════════════════════════════════════════════════════
def show_output_review(in_path, cols, data, output_data,
                       time_in, new_time, event_times_in,
                       n_events, total_inserted, _testing=False):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    n_ch = data.shape[1]
    chan_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_ch)]
    state = {"decision": None}
    leg_line_map = {}
    vis = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
    if not any(vis):
        vis = [True] + [False] * (len(cols) - 1)

    fig = plt.figure(figsize=(17, 9))
    plot_ax = fig.add_axes([0.08, 0.13, 0.88, 0.75])
    slide_ax = fig.add_axes([0.08, 0.04, 0.88, 0.035])

    bw, bh, bx = 0.055, 0.042, 0.004
    btn_dcdt = Button(fig.add_axes([bx, 0.83, bw, bh]), "DCDT",
                      color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg = Button(fig.add_axes([bx, 0.78, bw, bh]), "Strain",
                    color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.73, bw, bh]), "Pressure",
                       color="#ffe0b0", hovercolor="#ffc870")
    btn_all = Button(fig.add_axes([bx, 0.68, bw, bh]), "All on",
                     color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none = Button(fig.add_axes([bx, 0.63, bw, bh]), "All off",
                      color="#e8e8e8", hovercolor="#d0d0d0")
    btn_next = Button(fig.add_axes([bx, 0.55, bw, bh]), "Next\n20s",
                      color="#dde8ff", hovercolor="#bbd0ff")
    btn_prev = Button(fig.add_axes([bx, 0.50, bw, bh]), "Prev\n20s",
                      color="#dde8ff", hovercolor="#bbd0ff")
    btn_save = Button(fig.add_axes([bx, 0.44, bw, bh]), "Save",
                      color="#90ee90", hovercolor="#60dd60")
    btn_discard = Button(fig.add_axes([bx, 0.39, bw, bh]), "Discard",
                         color="#ffaaaa", hovercolor="#ff8888")
    for b in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none,
              btn_next, btn_prev, btn_save, btn_discard):
        b.label.set_fontsize(7)

    t_data_max = max(float(time_in[-1]), float(new_time[-1]))
    t_max = max(t_data_max - WINDOW_S, 0.01)
    slider = Slider(slide_ax, "Time (s)", 0.0, t_max,
                    valinit=0.0, color="steelblue")

    fig.suptitle(
        f"OUTPUT REVIEW — {os.path.basename(in_path)}  |  "
        f"{n_events} events repaired  (+{total_inserted} rows)  |  "
        f"Input {len(data):,} -> Output {len(output_data):,} rows\n"
        "——  INPUT (original)     - - -  OUTPUT (fixed)     "
        "same colour = same channel     green dashed = repair point     "
        "click legend to toggle channel\n"
        "NOTE: the OUTPUT trace drifts right of the INPUT trace over the "
        "file — that is the recovered (previously dropped) time.",
        fontsize=8.5, fontweight="bold",
    )

    def redraw(_=None):
        t0 = slider.val
        t1 = t0 + WINDOW_S
        mask_in = (time_in >= t0) & (time_in <= t1)
        mask_out = (new_time >= t0) & (new_time <= t1)

        plot_ax.cla()
        y_labels = []
        vis_idx = [i for i in range(n_ch) if vis[i]]

        for i in vis_idx:
            c = cols[i]
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

        for t_ev in event_times_in:
            if t0 - 1 <= t_ev <= t1 + 1:
                plot_ax.axvline(t_ev, color="green", lw=1.2,
                                ls="--", alpha=0.65, zorder=3)

        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("  /  ".join(y_labels) if y_labels else "Value",
                           fontsize=8)
        plot_ax.set_xlim(t0, t1)
        plot_ax.grid(True, alpha=0.25)

        leg_line_map.clear()
        if vis_idx:
            leg = plot_ax.legend(fontsize=6, ncol=3, loc="upper right")
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
            if group == "dcdt":
                vis[i] = c.startswith("DCDT_")
            elif group == "sg":
                vis[i] = c.startswith("SG_")
            elif group == "press":
                vis[i] = ("pressure" in c.lower() or c.startswith("volt_ch"))
            elif group == "all":
                vis[i] = True
            elif group == "none":
                vis[i] = False
        redraw()

    def on_legend_pick(event):
        ll = event.artist
        if ll not in leg_line_map:
            return
        vis[leg_line_map[ll]] = not vis[leg_line_map[ll]]
        redraw()

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _: set_group("dcdt"))
    btn_sg.on_clicked(lambda _: set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _: set_group("all"))
    btn_none.on_clicked(lambda _: set_group("none"))
    btn_next.on_clicked(lambda _: slider.set_val(min(slider.val + 20.0, t_max)))
    btn_prev.on_clicked(lambda _: slider.set_val(max(slider.val - 20.0, 0.0)))

    def on_save(_):
        state["decision"] = True
        plt.close(fig)

    def on_discard(_):
        state["decision"] = False
        plt.close(fig)

    btn_save.on_clicked(on_save)
    btn_discard.on_clicked(on_discard)
    fig.canvas.mpl_connect("pick_event", on_legend_pick)

    redraw()
    if _testing:                 # build/redraw only; used by automated tests
        plt.close(fig)
        return True
    plt.show()
    return state["decision"]


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITING
# ══════════════════════════════════════════════════════════════════════════════
def write_outputs(out, args, in_path, group_events, mode):
    out.to_csv(args.output, sep="\t", index=False, float_format="%.6f",
               lineterminator="\r\n")
    print(f"written: {args.output}  ({len(out)} rows)")

    if args.clean_output:
        flag_cols = [c for c in out.columns
                     if c.startswith("inserted_") or c.startswith("repaired_")]
        out.drop(columns=flag_cols).to_csv(
            args.clean_output, sep="\t", index=False, float_format="%.6f",
            lineterminator="\r\n")
        print(f"written: {args.clean_output}  (no flag columns)")

    if args.eventlog:
        rows = [(g, j, k) for g, ev in group_events.items() for j, k in ev]
        pd.DataFrame(rows, columns=["group", "jump_row", "k_dropped"]) \
            .to_csv(args.eventlog, index=False)
        print(f"written: {args.eventlog}")

    # sidecar repair log — permanent record of what was synthesised
    log_path = os.path.splitext(args.output)[0] + "_repairlog.txt"
    with open(log_path, "w") as f:
        f.write("REPAIR LOG — fix_bad_cycles.py\n")
        f.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Input     : {in_path}\n")
        f.write(f"Output    : {args.output}\n")
        f.write(f"Mode      : {mode}\n\n")
        for g, ev in group_events.items():
            ks = [k for _, k in ev]
            f.write(f"{g}: {len(ev)} events, k range "
                    f"{min(ks)}-{max(ks)}\n" if ev else f"{g}: 0 events\n")
        f.write(f"\n  {'group':>6}  {'jump row':>10}  {'time (s)':>10}  "
                f"{'k dropped':>9}\n")
        for g, ev in group_events.items():
            for j, k in ev:
                f.write(f"  {g:>6}  {j:>10,}  {j / SAMPLE_RATE:>10.3f}  "
                        f"{k:>9}\n")
    print(f"written: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=None,
                    help="input file; if omitted a file browser opens")
    ap.add_argument("-o", "--output", default=None,
                    help="output path; if omitted a folder picker opens on Save")
    ap.add_argument("--mode", choices=["insert", "replace", "drop"],
                    default="insert")
    ap.add_argument("--clean-output", default=None,
                    help="also write a copy WITHOUT the flag columns")
    ap.add_argument("--eventlog", default=None,
                    help="optional CSV path for the event log")
    ap.add_argument("--no-flags", action="store_true",
                    help="omit inserted_/repaired_ flag columns")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the interactive review plot (headless)")
    args = ap.parse_args()

    # ── File browser when no input given ─────────────────────────────────────
    in_path = args.input
    if in_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        in_path = filedialog.askopenfilename(
            title="Select trimmed input file",
            initialdir=os.path.dirname(os.path.abspath(__file__)),
            filetypes=[("Trimmed files", "*_trimmed.txt"),
                       ("Text files", "*.txt"), ("All files", "*.*")])
        root.destroy()
        if not in_path:
            sys.exit("No file selected.")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(in_path, sep="\t")
    cols = [c for c in df.columns if c != "time_s"]
    print(f"\nInput    : {os.path.basename(in_path)}")
    print(f"Rows     : {len(df):,}  ({(len(df) - 1) / SAMPLE_RATE:.2f} s)")
    print(f"Channels : {len(cols)}")

    for g, s in GROUPS.items():
        if s["ref"] not in df.columns:
            sys.exit(f"reference channel {s['ref']} not found — edit GROUPS "
                     f"at the top of this script for this rig configuration")

    # ── Detect ────────────────────────────────────────────────────────────────
    group_events = {g: detect_events(df[s["ref"]].to_numpy())
                    for g, s in GROUPS.items()}
    for g, ev in group_events.items():
        ks = [k for _, k in ev]
        print(f"{g}: {len(ev)} events, k (samples dropped) range "
              f"{min(ks)}-{max(ks)}" if ev else f"{g}: 0 events")

    # ── Repair ────────────────────────────────────────────────────────────────
    data_in = df[cols].to_numpy(float)
    time_in = np.arange(len(df)) / SAMPLE_RATE

    if args.mode == "insert":
        out = repair_insert(df, group_events)
        if args.no_flags:
            out = out.drop(columns=[c for c in out.columns
                                    if c.startswith("inserted_")])
        print(f"insert mode: {len(df)} -> {len(out)} rows "
              f"({len(out) / SAMPLE_RATE:.1f} s reconstructed test time)")
    elif args.mode == "replace":
        group_cols = {g: [c for c in df.columns if c.startswith(s["prefix"])]
                      for g, s in GROUPS.items()}
        flags = repair_replace(df, group_events, group_cols)
        out = df
        if not args.no_flags:
            for g in GROUPS:
                out[f"repaired_{g}"] = flags[g]
    else:
        out, log = repair_drop(df, group_events)
        print(f"dropped {sum(L for _, L, _ in log)} rows across {len(log)} "
              f"events (note: SG channels keep a 2-sample phase kink)")

    validate(out, "after repair")

    total_inserted = len(out) - len(df)
    new_time = out["time_s"].to_numpy()
    data_out = out[cols].to_numpy(float)
    event_times_in = [j / SAMPLE_RATE for j, _ in group_events["DCDT"]]

    # ── Review plot ───────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\nOpening output review plot...")
        save_it = show_output_review(in_path, cols, data_in, data_out,
                                     time_in, new_time, event_times_in,
                                     len(group_events["DCDT"]),
                                     total_inserted)
        if not save_it:
            sys.exit("\nDiscarded — no file written.")

    # ── Resolve output paths ──────────────────────────────────────────────────
    if args.output is None:
        stem = os.path.splitext(os.path.basename(in_path))[0]
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        save_dir = filedialog.askdirectory(
            title="Select folder to save fixed file",
            initialdir=os.path.dirname(os.path.abspath(in_path)))
        root.destroy()
        if not save_dir:
            sys.exit("\nSave cancelled — no file written.")
        args.output = os.path.join(save_dir, stem + "_fixed.txt")
        if args.clean_output is None:
            args.clean_output = os.path.join(save_dir, stem + "_fixed_clean.txt")
        if args.eventlog is None:
            args.eventlog = os.path.join(save_dir, stem + "_events.csv")

    write_outputs(out, args, in_path, group_events, args.mode)
    print("Done.")


if __name__ == "__main__":
    main()