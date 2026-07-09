"""
postprocess_peaks.py — Dropout repair via zero-crossing anchoring,
multi-channel voting, and rolling-template validation.

PROBLEM
  The DAQ occasionally drops N samples (typically 7-9), leaving a
  "compressed" cycle of SPC-N rows instead of SPC=16.  Drops remove whole
  rows, so every channel loses the same instants — detection is per
  channel (voted), but repair is applied to full rows so all channels
  stay aligned row-by-row.

WHY ZERO-CROSSINGS (not peaks)
  A flat-topped signal has peak position uncertainty of +/-2-3 rows, so
  peak-to-peak gaps naturally range 10-22 rows.  The upward zero-crossing
  occurs on the steepest slope, so its position is precise to < 0.5 rows.
  A dropout of N samples gives gap = SPC - N versus a clean 16.0 +/- 0.5.

PIPELINE
  1. File browser   — select trimmed input file
  2. Detect         — per-DCDT-channel upward zero-crossings; gaps
                      < SPC - GAP_TOLERANCE flag a compressed cycle, and
                      gaps of SPC+GAP_TOLERANCE .. 2*SPC-GAP_TOLERANCE flag
                      a dropout that swallowed the crossing itself (the
                      bad region then spans 2 nominal cycles)
  3. Vote & merge   — detections within MERGE_WINDOW rows are one event;
                      >= MIN_CH_CONFIRM channels must agree
  4. Cross-check    — the aggregate (mean-DCDT) signal must also show the
                      event; unmatched events are logged, not repaired
  5. Template       — rolling median template (last TEMPLATE_N clean
                      cycles per channel, shape-normalised) validates
                      donor cycles before they are pasted
  6. Repair         — erase bad cycle(s) + 1 trailing good cycle, paste
                      validated clean donor cycles (full rows, all
                      channels at once, descending row order)
  7. Verify         — re-run gap detection + template scan on the OUTPUT
                      and report PASS/FAIL before anything is saved
  8. Review plot    — INPUT (solid) vs OUTPUT (dashed), green = repairs
  9. Save           — aligned data file + sidecar *_repairlog.txt

Usage:
    python postprocess_peaks.py
"""

import os
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16     # Hz
CYCLE_HZ        = 1.0    # Hz
RAMP_SECONDS    = 5.0    # seconds to skip at start (not cyclic)
GAP_TOLERANCE   = 3      # flag gaps < SPC - GAP_TOLERANCE  (i.e. < 13 for SPC=16)
MIN_CH_CONFIRM  = 2      # DCDT channels that must agree on an event
MERGE_WINDOW    = 6      # rows: detections within this window -> single event
MAX_MISS        = 12     # max recoverable dropped samples per single event
CLEAN_GAP_TOL   = 1.0    # |gap - SPC| <= this  ->  cycle counts as clean
TEMPLATE_N      = 10     # clean cycles in the rolling median template
MIN_TEMPLATE    = 3      # cycles needed before template scoring starts
SHAPE_ERR_MAX   = 0.18   # normalised-RMS shape error above which a cycle fails
DONOR_MATCH_TOL = 3      # rows: donor cycle must sit within this of a
                         # template-verified clean cycle start

WINDOW_S  = 20.0         # seconds visible in review plot
MARKER_MS = 4
LINE_W    = 0.9

# ── Derived ───────────────────────────────────────────────────────────────────
SPC       = int(round(SAMPLE_RATE / CYCLE_HZ))    # rows per cycle = 16
RAMP_ROWS = int(RAMP_SECONDS * SAMPLE_RATE)       # rows to skip   = 80
GAP_MAX   = SPC - GAP_TOLERANCE                   # flag gaps shorter than this

# ── 24-colour palette ─────────────────────────────────────────────────────────
_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def find_upcrossings(signal, level):
    """
    Return fractional row indices where signal crosses 'level' going upward.
    Linear interpolation gives sub-sample precision (< 0.5 rows on the
    steep slope of the waveform).
    """
    out = []
    for i in range(len(signal) - 1):
        if signal[i] < level <= signal[i + 1]:
            frac = (level - signal[i]) / (signal[i + 1] - signal[i])
            out.append(i + frac)
    return np.array(out)


def midpoint_level(sig):
    """Robust oscillation midpoint: mean of 5th/95th percentiles."""
    return (float(np.percentile(sig, 5)) + float(np.percentile(sig, 95))) / 2.0


def segment_cycles(sig, ramp_rows=None):
    """Fractional cycle-start rows (upward midpoint crossings) after the ramp."""
    if ramp_rows is None:
        ramp_rows = RAMP_ROWS
    cyc = sig[ramp_rows:]
    if len(cyc) < 2 * SPC:
        return np.array([])
    level = midpoint_level(cyc)
    return find_upcrossings(cyc, level) + ramp_rows


def extract_cycle(sig, start_frac):
    """One cycle (SPC points) starting at a fractional row, via interpolation.
    Fractional sampling removes the +/-0.5-row phase jitter that integer
    rounding would inject into the shape score.
    Interpolates over a small LOCAL window, not the full signal — this is
    called once per channel per cycle (tens of thousands of times on a
    long file), so rebuilding a full-length index array each call would
    make the whole scan scale as O(cycles * channels * N)."""
    idx    = start_frac + np.arange(SPC)
    lo     = max(int(np.floor(start_frac)) - 1, 0)
    hi     = min(int(np.ceil(idx[-1])) + 2, len(sig))
    return np.interp(idx, np.arange(lo, hi), sig[lo:hi])


def norm_shape(seg):
    """Shape-only normalisation: zero-mean, unit peak-to-peak.
    Makes the template score immune to amplitude/offset drift so one
    threshold works for every channel over the whole test."""
    seg = seg - seg.mean()
    ptp = seg.max() - seg.min()
    return seg / ptp if ptp > 1e-12 else seg


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION  — per-channel crossing gaps, then multi-channel voting
# ══════════════════════════════════════════════════════════════════════════════
def classify_gap(gap):
    """
    Classify a crossing-to-crossing gap.  Returns (span, n_miss):
      span 1 — compressed cycle (dropout between two surviving crossings)
      span 2 — dropout swallowed the crossing itself, so two nominal
               cycles merged into one long gap
      span 0 — clean / unclassifiable
    """
    if 0 < gap < GAP_MAX:
        return 1, int(round(SPC - gap))
    if SPC + GAP_TOLERANCE < gap < 2 * SPC - GAP_TOLERANCE:
        return 2, int(round(2 * SPC - gap))
    return 0, 0


def detect_events(data, det_idx, cols):
    """
    Per-channel zero-crossing gap detection + vote merge.
    Returns (merged_events, vote_counts, n_raw) where each merged event is
    (cycle_start, cycle_end, n_missing, span, [channels], n_votes).
    """
    raw_events = []
    for ch_i in det_idx:
        xings = segment_cycles(data[:, ch_i])
        for j in range(1, len(xings)):
            span, n_miss = classify_gap(xings[j] - xings[j - 1])
            if span and n_miss > 0:
                raw_events.append((int(round(xings[j - 1])),
                                   int(round(xings[j])),
                                   n_miss, span, cols[ch_i]))
    raw_events.sort(key=lambda x: x[0])

    merged, vote_counts = [], {}
    i = 0
    while i < len(raw_events):
        group = [raw_events[i]]
        j = i + 1
        while (j < len(raw_events)
               and raw_events[j][0] - raw_events[i][0] <= MERGE_WINDOW):
            group.append(raw_events[j])
            j += 1

        # The same dropout can look compressed (span 1) to one channel and
        # over-long (span 2) to another when the crossing itself was dropped.
        # Medians across mixed types produce a hybrid event whose start/end
        # match no real crossing pair — so merge only the majority-span
        # subset; ties prefer span 1 (smaller, simpler repair).
        spans = [e[3] for e in group]
        span  = min(set(spans), key=lambda s: (-spans.count(s), s))
        sub   = [e for e in group if e[3] == span]

        cycle_start = int(np.median([e[0] for e in sub]))
        cycle_end   = int(np.median([e[1] for e in sub]))
        # n_miss from the measured gap keeps event geometry self-consistent
        n_miss      = span * SPC - (cycle_end - cycle_start)
        ch_names    = sorted({e[4] for e in sub})
        n_votes     = len(ch_names)

        vote_counts[n_votes] = vote_counts.get(n_votes, 0) + 1
        if n_votes >= MIN_CH_CONFIRM and n_miss > 0:
            merged.append((cycle_start, cycle_end, n_miss, span,
                           ch_names, n_votes))
        i = j
    return merged, vote_counts, len(raw_events)


def aggregate_compressed(data, det_idx):
    """Dropout gaps seen by the mean-DCDT signal: (start, end, n_miss, span)."""
    sig = data[:, det_idx].mean(axis=1)
    x   = segment_cycles(sig)
    out = []
    for j in range(1, len(x)):
        span, n_miss = classify_gap(x[j] - x[j - 1])
        if span and n_miss > 0:
            out.append((int(round(x[j - 1])), int(round(x[j])),
                        n_miss, span))
    return out


def cross_check(merged, agg_events):
    """
    A real dropout removes rows from every channel, so it MUST also appear
    in the aggregate signal.  Voted events without an aggregate match are
    dropped (logged); aggregate events no vote confirmed are reported too.
    Matching compares row RANGES, not start positions: the same dropout
    can anchor one cycle earlier in the signal whose crossing it swallowed.
    Returns (confirmed, dropped_no_agg, agg_unconfirmed).
    """
    def ranges_match(cs, ce, a0, a1):
        separation = max(a0 - ce, cs - a1)   # <= 0 means the ranges overlap
        return separation <= MERGE_WINDOW

    confirmed, dropped = [], []
    for ev in merged:
        if any(ranges_match(ev[0], ev[1], a[0], a[1]) for a in agg_events):
            confirmed.append(ev)
        else:
            dropped.append(ev)
    agg_unconfirmed = [a for a in agg_events
                       if not any(ranges_match(e[0], e[1], a[0], a[1])
                                  for e in merged)]
    return confirmed, dropped, agg_unconfirmed


def dedup_overlaps(events):
    """
    Two detections whose row ranges overlap describe the SAME dropout —
    e.g. some channels see a compressed gap while others (whose crossing
    was inside the dropped block) see an over-long gap starting one cycle
    earlier.  Keep the detection with more channel votes.
    Genuinely consecutive bad cycles touch but do not overlap, so they
    survive this filter and are handled by plan_repairs().
    """
    events = sorted(events, key=lambda e: e[0])
    out = []
    for ev in events:
        if out and ev[0] <= out[-1][1] - 3:      # overlaps previous range
            if ev[5] > out[-1][5]:               # more votes wins
                out[-1] = ev
        else:
            out.append(ev)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDATION  — rolling median template per channel
#   Every gap-clean cycle is scored against the median of the last
#   TEMPLATE_N accepted cycles (shape-normalised, so amplitude drift over
#   the test does not matter).  A cycle is "verified clean" when at most
#   a minority of detection channels fail the shape threshold.  Only
#   verified cycles may be used as repair donors.
# ══════════════════════════════════════════════════════════════════════════════
def build_verified_starts(data, det_idx):
    """Returns (sorted int array of verified clean-cycle starts, all scores)."""
    sig = data[:, det_idx].mean(axis=1)
    x   = segment_cycles(sig)
    templates = {ch: deque(maxlen=TEMPLATE_N) for ch in det_idx}
    verified, scores = [], []

    for j in range(1, len(x)):
        gap = x[j] - x[j - 1]
        if abs(gap - SPC) > CLEAN_GAP_TOL:
            continue                       # not a full-length cycle
        start  = x[j - 1]
        n_fail = 0
        for ch in det_idx:
            seg = norm_shape(extract_cycle(data[:, ch], start))
            dq  = templates[ch]
            ok  = True
            if len(dq) >= MIN_TEMPLATE:
                tmpl  = np.median(np.stack(list(dq)), axis=0)
                score = float(np.sqrt(np.mean((seg - tmpl) ** 2)))
                scores.append(score)
                ok = score <= SHAPE_ERR_MAX
                if not ok:
                    n_fail += 1
            if ok:
                dq.append(seg)             # failing cycles never enter template
        if n_fail <= len(det_idx) // 2:
            verified.append(int(round(start)))

    return np.array(sorted(verified), dtype=int), scores


# ══════════════════════════════════════════════════════════════════════════════
# REPAIR  — replace compressed cycle(s) with validated clean donor cycles
#
#   Single bad cycle  (n_ref = 2):
#     Erase  : bad cycle + 1 good after  ->  2*SPC - n_miss  rows removed
#     Insert : 2 previous clean cycles   ->  2*SPC           rows added
#     Net    : +n_miss
#   Two consecutive bad cycles (n_ref = 3): erase 2 bad + 1 good,
#     insert 3 clean, net +(n1+n2).
#   The extra good cycle absorbs the +/-1-row uncertainty of the bad
#   cycle's end boundary so the splice joins at equivalent phase points.
# ══════════════════════════════════════════════════════════════════════════════
def plan_repairs(merged):
    """Merge consecutive bad-cycle pairs into compound events.
    n_ref = total nominal cycles the bad region spans + 1 trailing good
    cycle, so the paste always replaces exactly what is erased + n_miss.
    Returns list of (cycle_start, replace_end, n_miss_total, n_ref)."""
    sorted_asc = sorted(merged, key=lambda x: x[0])
    compound   = []
    i = 0
    while i < len(sorted_asc):
        cs, ce, _, span, _, _ = sorted_asc[i]
        if (i + 1 < len(sorted_asc)
                and abs(sorted_asc[i + 1][0] - ce) <= MERGE_WINDOW):
            _, ce2, _, span2, _, _ = sorted_asc[i + 1]
            n_ref, replace_end = span + span2 + 1, ce2 + SPC
            i += 2
        else:
            n_ref, replace_end = span + 1, ce + SPC
            i += 1
        # rows recovered = paste minus erase, from the actual splice
        # geometry — guarantees the row ledger is exact
        nm_eff = n_ref * SPC - (replace_end - cs)
        compound.append((cs, replace_end, nm_eff, n_ref))
    return compound


def apply_repairs(data, compound, verified_starts, bad_starts):
    """
    Apply repairs in DESCENDING row order (earlier indices stay valid).
    Donor cycles must each sit within DONOR_MATCH_TOL rows of a
    template-verified clean cycle; otherwise walk back one cycle at a time.
    Returns (output_data, applied, skipped).
    """
    out = data.copy()
    applied, skipped = [], []

    def donors_ok(ref_start, n_ref):
        if len(verified_starts) == 0:
            # Fallback when no template could be built: only avoid known
            # bad cycles (the pre-template behaviour).
            return not any((ref_start + k * SPC) in bad_starts
                           for k in range(n_ref))
        for k in range(n_ref):
            want = ref_start + k * SPC
            pos  = int(np.searchsorted(verified_starts, want))
            best = 10 ** 9
            if pos > 0:
                best = min(best, abs(int(verified_starts[pos - 1]) - want))
            if pos < len(verified_starts):
                best = min(best, abs(int(verified_starts[pos]) - want))
            if best > DONOR_MATCH_TOL:
                return False
        return True

    for cs, replace_end, nm, n_ref in sorted(compound, key=lambda x: x[0],
                                             reverse=True):
        if replace_end > len(out):
            skipped.append((cs, nm, "extends past end of file"))
            continue
        if cs < RAMP_ROWS:
            skipped.append((cs, nm, "inside ramp region"))
            continue

        ref = cs - n_ref * SPC
        while ref >= RAMP_ROWS and not donors_ok(ref, n_ref):
            ref -= SPC
        if ref < RAMP_ROWS:
            skipped.append((cs, nm, "no verified donor cycles before event"))
            continue

        donor = out[ref: ref + n_ref * SPC].copy()
        out = np.vstack([
            out[:cs],
            donor,                # n_ref * SPC rows, all channels
            out[replace_end:]     # skip bad cycle(s) + trailing good cycle
        ])
        applied.append({"row": cs, "time_s": cs / SAMPLE_RATE,
                        "n_miss": nm, "n_ref": n_ref, "donor_row": ref})
    return out, applied, skipped


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION  — prove the repair worked before saving
#   Re-runs gap detection AND the rolling-template shape scan on a dataset.
#   The output must show zero compressed cycles and zero shape failures.
# ══════════════════════════════════════════════════════════════════════════════
def verify_dataset(arr, det_idx, label):
    sig  = arr[:, det_idx].mean(axis=1)
    x    = segment_cycles(sig)
    gaps = np.diff(x) if len(x) > 1 else np.array([])
    rep  = {
        "label":        label,
        "n_cycles":     int(len(gaps)),
        "n_compressed": int(np.sum(gaps < GAP_MAX)) if len(gaps) else 0,
        "n_long":       int(np.sum(gaps > SPC + GAP_TOLERANCE)) if len(gaps) else 0,
    }
    templates = {ch: deque(maxlen=TEMPLATE_N) for ch in det_idx}
    fails, worst = 0, 0.0
    for j in range(1, len(x)):
        if abs((x[j] - x[j - 1]) - SPC) > CLEAN_GAP_TOL:
            continue
        for ch in det_idx:
            seg = norm_shape(extract_cycle(arr[:, ch], x[j - 1]))
            dq  = templates[ch]
            if len(dq) >= MIN_TEMPLATE:
                tmpl  = np.median(np.stack(list(dq)), axis=0)
                score = float(np.sqrt(np.mean((seg - tmpl) ** 2)))
                worst = max(worst, score)
                if score > SHAPE_ERR_MAX:
                    fails += 1
                    continue
            dq.append(seg)
    rep["shape_fails"]       = fails
    rep["worst_shape_score"] = round(worst, 4)
    return rep


def verification_lines(rep):
    return [
        f"  [{rep['label']}]",
        f"    cycles found        : {rep['n_cycles']}",
        f"    compressed (<{GAP_MAX} rows): {rep['n_compressed']}",
        f"    over-long  (>{SPC + GAP_TOLERANCE} rows): {rep['n_long']}",
        f"    shape failures      : {rep['shape_fails']}"
        f"  (worst score {rep['worst_shape_score']},"
        f" threshold {SHAPE_ERR_MAX})",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT REVIEW PLOT
#   INPUT  = solid line + dots    OUTPUT = dashed line + dots   (same colour)
#   Green dashed verticals = repair locations
# ══════════════════════════════════════════════════════════════════════════════
def show_output_review(in_path, cols, chan_colors, data, output_data,
                       time_in, new_time, insert_times_in,
                       n_events, total_inserted):
    n_ch         = data.shape[1]
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
        f"{n_events} repairs  (+{total_inserted} rows)  |  "
        f"Input {len(data):,} → Output {len(output_data):,} rows\n"
        "——  INPUT (original)     - - -  OUTPUT (fixed)     "
        "same colour = same channel     green dashed = repair point     "
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — interactive flow
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── File browser ──────────────────────────────────────────────────────────
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

    # ── Load ──────────────────────────────────────────────────────────────────
    df      = pd.read_csv(in_path, sep="\t")
    cols    = [c for c in df.columns if c != "time_s"]
    data    = df[cols].values.astype(float)
    N, n_ch = data.shape
    time_in = np.arange(0, N) / SAMPLE_RATE

    det_idx = [i for i, c in enumerate(cols) if c.startswith("DCDT_")]
    if not det_idx:
        det_idx = list(range(min(4, n_ch)))   # fallback if no DCDT_ columns

    chan_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_ch)]

    print(f"\nInput      : {os.path.basename(in_path)}")
    print(f"Rows       : {N:,}  ({time_in[-1]:.2f} s)")
    print(f"Channels   : {n_ch}  |  detection on: {[cols[i] for i in det_idx]}")
    print(f"SPC        : {SPC} rows/cycle  |  flagging gaps < {GAP_MAX} rows")

    # ── Detect (per channel) + vote ───────────────────────────────────────────
    merged_events, vote_counts, n_raw = detect_events(data, det_idx, cols)

    print(f"\nRaw detections : {n_raw}  "
          f"({len(det_idx)} DCDT channels, upward zero-crossings)")
    print(f"Group vote distribution (MIN_CH_CONFIRM={MIN_CH_CONFIRM}):")
    for v in sorted(vote_counts):
        flag = "  <- accepted" if v >= MIN_CH_CONFIRM else "  <- rejected"
        print(f"  {v} channel(s) agreed : {vote_counts[v]:>6} groups{flag}")

    # ── Cross-check against the aggregate signal ──────────────────────────────
    agg_events = aggregate_compressed(data, det_idx)
    merged_events, no_agg_match, agg_unconfirmed = cross_check(merged_events,
                                                               agg_events)
    merged_events = dedup_overlaps(merged_events)
    if no_agg_match:
        print(f"\nWARNING: {len(no_agg_match)} voted event(s) NOT visible in "
              f"the aggregate signal — dropped (rows: "
              f"{[e[0] for e in no_agg_match]})")
    if agg_unconfirmed:
        print(f"WARNING: {len(agg_unconfirmed)} aggregate compressed cycle(s) "
              f"not confirmed by channel voting — NOT repaired (rows: "
              f"{[a[0] for a in agg_unconfirmed]})")

    # ── Reject events beyond repair ───────────────────────────────────────────
    too_long = [e for e in merged_events if e[2] > MAX_MISS]
    merged_events = [e for e in merged_events if e[2] <= MAX_MISS]
    if too_long:
        print(f"\nWARNING: {len(too_long)} event(s) exceed MAX_MISS={MAX_MISS} "
              f"dropped samples — NOT repaired (rows: "
              f"{[e[0] for e in too_long]})")

    print(f"\nConfirmed events : {len(merged_events)}  "
          f"(voted by {MIN_CH_CONFIRM}+ channels AND seen in aggregate)")
    print()
    print(f"  {'#':>4}  {'Cycle start':>11}  {'Time (s)':>10}  "
          f"{'N recover':>10}  {'Span':>4}  {'Votes':>6}  Channels")
    print(f"  {'-'*4}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*4}  {'-'*6}  {'-'*28}")
    for k, (cs, ce, nm, sp, chs, nv) in enumerate(merged_events):
        print(f"  {k+1:>4}  {cs:>11,}  {cs / SAMPLE_RATE:>10.3f}  "
              f"{nm:>10}  {sp:>4}  {nv:>6}  {', '.join(chs)}")

    if not merged_events:
        print("\nNo dropout events detected — output would be identical "
              "to input.")
        raise SystemExit(0)

    # ── Template validation of clean cycles ───────────────────────────────────
    verified_starts, scores = build_verified_starts(data, det_idx)
    if scores:
        s = np.array(scores)
        print(f"\nTemplate scores (clean cycles): median {np.median(s):.4f}, "
              f"95th pct {np.percentile(s, 95):.4f}, max {s.max():.4f}  "
              f"(threshold {SHAPE_ERR_MAX})")
        if np.percentile(s, 95) > 0.8 * SHAPE_ERR_MAX:
            print("  NOTE: scores are close to the threshold — consider "
                  "raising SHAPE_ERR_MAX if good donors are being rejected.")
    print(f"Verified clean cycles (donor pool): {len(verified_starts)}")

    # ── Plan + apply repairs ──────────────────────────────────────────────────
    compound = plan_repairs(merged_events)
    n_single = sum(1 for ev in compound if ev[3] == 2)
    n_double = sum(1 for ev in compound if ev[3] == 3)
    print(f"\nFix plan   : {len(compound)} repairs  "
          f"— {n_single} single bad cycle,  {n_double} double consecutive")

    bad_starts = {e[0] for e in merged_events}
    output_data, applied, skipped = apply_repairs(data, compound,
                                                  verified_starts, bad_starts)
    for cs, nm, reason in skipped:
        print(f"  SKIPPED repair at row {cs:,}: {reason}")

    M              = len(output_data)
    new_time       = np.arange(0, M) / SAMPLE_RATE
    total_inserted = M - N
    expected       = sum(a["n_miss"] for a in applied)

    print(f"\n{'-'*54}")
    print(f"  Detected   : {len(merged_events)} bad cycles  "
          f"->  {len(applied)} repairs applied, {len(skipped)} skipped")
    print(f"  Inserted   : +{total_inserted} rows  "
          f"({total_inserted / SAMPLE_RATE:.4f} s recovered)"
          f"  |  expected +{expected}"
          f"  {'OK' if total_inserted == expected else '<-- MISMATCH'}")
    print(f"  Input rows : {N:,}  ({time_in[-1]:.4f} s)")
    print(f"  Output rows: {M:,}  ({new_time[-1]:.4f} s)")
    print(f"{'-'*54}")

    # ── Verification pass ─────────────────────────────────────────────────────
    rep_in  = verify_dataset(data,        det_idx, "INPUT  (before repair)")
    rep_out = verify_dataset(output_data, det_idx, "OUTPUT (after repair)")
    print("\nVERIFICATION")
    for ln in verification_lines(rep_in) + verification_lines(rep_out):
        print(ln)
    verify_pass = (rep_out["n_compressed"] == 0
                   and rep_out["n_long"] == 0
                   and total_inserted == expected)
    print(f"\n  RESULT: {'PASS — output is clean' if verify_pass else 'FAIL — inspect the review plot carefully before saving'}")

    # ── Review plot ───────────────────────────────────────────────────────────
    insert_times_in = [a["time_s"] for a in applied]
    print("\nOpening output review plot...")
    save_it = show_output_review(in_path, cols, chan_colors, data, output_data,
                                 time_in, new_time, insert_times_in,
                                 len(applied), total_inserted)
    if not save_it:
        print("\nDiscarded — no file written.")
        raise SystemExit(0)

    # ── Save — folder picker, filename auto-generated ─────────────────────────
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

    # ── Sidecar repair log — repaired cycles are fabricated data; this file
    #    is the permanent record of exactly which rows were synthesised ────────
    log_path = os.path.splitext(out_path)[0] + "_repairlog.txt"
    with open(log_path, "w") as f:
        f.write("REPAIR LOG — postprocess_peaks.py\n")
        f.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Input     : {in_path}\n")
        f.write(f"Output    : {out_path}\n\n")
        f.write(f"Config    : SAMPLE_RATE={SAMPLE_RATE}  SPC={SPC}  "
                f"GAP_TOLERANCE={GAP_TOLERANCE}  MIN_CH_CONFIRM={MIN_CH_CONFIRM}  "
                f"MERGE_WINDOW={MERGE_WINDOW}\n")
        f.write(f"            TEMPLATE_N={TEMPLATE_N}  "
                f"SHAPE_ERR_MAX={SHAPE_ERR_MAX}  MAX_MISS={MAX_MISS}\n\n")
        f.write(f"Rows      : input {N:,} -> output {M:,}  "
                f"(+{total_inserted} rows, "
                f"{total_inserted / SAMPLE_RATE:.4f} s recovered)\n\n")
        f.write("REPAIRS APPLIED (row/time in INPUT coordinates; the pasted "
                "rows are COPIES of the donor cycles, not measured data)\n")
        f.write(f"  {'#':>3}  {'input row':>10}  {'time (s)':>9}  "
                f"{'n_miss':>6}  {'donor rows':>18}\n")
        for k, a in enumerate(sorted(applied, key=lambda x: x['row'])):
            d0, d1 = a["donor_row"], a["donor_row"] + a["n_ref"] * SPC
            f.write(f"  {k+1:>3}  {a['row']:>10,}  {a['time_s']:>9.3f}  "
                    f"{a['n_miss']:>6}  {d0:>8,} - {d1:<8,}\n")
        if skipped:
            f.write("\nSKIPPED (NOT repaired)\n")
            for cs, nm, reason in skipped:
                f.write(f"  row {cs:,} (n_miss={nm}): {reason}\n")
        if agg_unconfirmed:
            f.write("\nUNCONFIRMED aggregate detections (NOT repaired)\n")
            for a0, a1, nm, sp in agg_unconfirmed:
                f.write(f"  rows {a0:,}-{a1:,} (n_miss~{nm})\n")
        if too_long:
            f.write(f"\nEVENTS EXCEEDING MAX_MISS={MAX_MISS} (NOT repaired)\n")
            for cs, ce, nm, sp, chs, nv in too_long:
                f.write(f"  row {cs:,} (n_miss={nm})\n")
        f.write("\nVERIFICATION\n")
        for ln in verification_lines(rep_in) + verification_lines(rep_out):
            f.write(ln + "\n")
        f.write(f"\nRESULT: {'PASS' if verify_pass else 'FAIL'}\n")

    print(f"Done — {M:,} rows written.")
    print(f"  Time : {new_time[0]:.4f} s -> {new_time[-1]:.4f} s")
    print(f"  Log  : {log_path}")


if __name__ == "__main__":
    main()
