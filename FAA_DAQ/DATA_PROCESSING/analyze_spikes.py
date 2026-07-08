"""
Spike interval analysis — per sensor group (DCDT, Pressure, Strain).

Groups are detected automatically from column names:
  DCDT     : columns starting with "DCDT_"
  Pressure : columns containing "pressure" or starting with "volt_ch"
  Strain   : columns starting with "SG_"

Flat / inactive channels (pore water pressure sensors with no water, etc.)
are excluded automatically: a channel is "active" only if its MAD is at
least MIN_ACTIVE_MAD_FRAC of the group's maximum MAD.

Usage:
    python analyze_spikes.py                    # most recent session files
    python analyze_spikes.py 20260508_131204    # specific session timestamp
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob

# ── Configuration ─────────────────────────────────────────────────────────────
SPIKE_ZSCORE        = 8.0   # a channel spikes if its jump > this × its own MAD
FRAC_DCDT           = 0.50  # fraction of ACTIVE DCDT channels needed
FRAC_PRESSURE       = 0.50  # fraction of ACTIVE pressure channels needed
FRAC_STRAIN         = 0.50  # fraction of ACTIVE strain channels needed
MIN_ACTIVE_MAD_FRAC = 0.05  # channels with MAD < 5% of group max are excluded
MERGE_SAMPLES       = 3     # merge events within this many samples
# 'any'  → spike event if ANY group exceeds its threshold (sensitive)
# 'all'  → spike event if ALL groups exceed their threshold (conservative)
SPIKE_LOGIC         = 'any'

# ── Resolve files ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    ts        = sys.argv[1]
    raw_path  = f"data_raw_{ts}.txt"
    proc_path = f"data_processed_{ts}.txt"
else:
    raw_files  = sorted(glob("data_raw_*.txt"),       key=os.path.getmtime)
    proc_files = sorted(glob("data_processed_*.txt"), key=os.path.getmtime)
    if not raw_files or not proc_files:
        raise FileNotFoundError("No data_raw_*.txt / data_processed_*.txt found.")
    raw_path  = raw_files[-1]
    proc_path = proc_files[-1]

print(f"Raw file      : {raw_path}")
print(f"Processed file: {proc_path}")

# ── Column group detection ────────────────────────────────────────────────────
def get_groups(cols):
    dcdt   = [c for c in cols if c.startswith("DCDT_")]
    strain = [c for c in cols if c.startswith("SG_")]
    press  = [c for c in cols if "pressure" in c.lower() or c.startswith("volt_ch")]
    other  = [c for c in cols if c not in dcdt + strain + press]
    return {"DCDT": dcdt, "Pressure": press, "Strain": strain, "Other": other}

# ── Per-group spike detection ─────────────────────────────────────────────────
def group_spike_analysis(df, label):
    time  = df["time_s"].values
    cols  = [c for c in df.columns if c != "time_s"]
    data  = df[cols].values
    dt    = np.median(np.diff(time))
    N     = len(time)

    diffs = np.abs(np.diff(data, axis=0))          # (N-1, n_ch)
    mad   = np.median(diffs, axis=0)               # per-channel MAD

    groups  = get_groups(cols)
    results = {}

    print(f"\n{'='*65}")
    print(f"  {label}   |   {N} samples   {1/dt:.2f} Hz   {time[-1]:.1f} s")
    print(f"{'='*65}")

    group_spike_masks = {}   # bool (N-1,) per group

    for gname, gcols in groups.items():
        if not gcols:
            continue
        idx = [cols.index(c) for c in gcols]

        g_mad  = mad[idx]
        g_max_mad = g_mad.max() if g_mad.max() > 0 else 1e-12
        active = g_mad >= MIN_ACTIVE_MAD_FRAC * g_max_mad
        n_active = active.sum()

        if n_active == 0:
            print(f"\n  {gname:10s}: all channels flat/inactive — skipped")
            group_spike_masks[gname] = np.zeros(N - 1, dtype=bool)
            continue

        thresh_count = max(1, int(np.ceil({
            "DCDT":     FRAC_DCDT,
            "Pressure": FRAC_PRESSURE,
            "Strain":   FRAC_STRAIN,
        }.get(gname, 0.5) * n_active)))

        g_diffs = diffs[:, idx]
        g_mad_safe = np.where(active, g_mad, np.inf)   # inactive → never spike
        z = g_diffs / g_mad_safe[np.newaxis, :]
        ch_spike   = z > SPIKE_ZSCORE                  # (N-1, n_group_ch)
        n_spiking  = ch_spike.sum(axis=1)              # (N-1,)
        group_mask = n_spiking >= thresh_count

        group_spike_masks[gname] = group_mask

        # Hit rate per channel (only over samples where group triggered)
        event_rows = np.where(group_mask)[0]
        hit_rate = ch_spike[event_rows].mean(axis=0) * 100 if len(event_rows) > 0 else np.zeros(len(gcols))

        print(f"\n  {gname:10s}: {len(gcols)} channels total | "
              f"{n_active} active | threshold = {thresh_count} channels "
              f"({thresh_count/n_active*100:.0f}% of active)")
        print(f"  {'Channel':<34} {'MAD':>10}  {'Active':>8}  {'Hit rate':>10}")
        print(f"  {'-'*66}")
        for k, c in enumerate(gcols):
            active_str = "YES" if active[k] else "flat"
            print(f"  {c:<34} {g_mad[k]:>10.6f}  {active_str:>8}  {hit_rate[k]:>9.0f}%")

        results[gname] = {
            "cols": gcols, "idx": idx,
            "mad": g_mad, "active": active,
            "n_spiking": n_spiking, "thresh_count": thresh_count,
            "group_mask": group_mask, "hit_rate": hit_rate,
        }

    # ── Combined event detection ──────────────────────────────────────────────
    active_groups = [g for g in ["DCDT", "Pressure", "Strain"] if g in results]
    masks = [group_spike_masks[g] for g in active_groups]

    if SPIKE_LOGIC == 'all':
        combined = np.ones(N - 1, dtype=bool)
        for m in masks:
            combined &= m
    else:  # 'any'
        combined = np.zeros(N - 1, dtype=bool)
        for m in masks:
            combined |= m

    spike_idx   = np.where(combined)[0]
    spike_times = time[spike_idx + 1]

    # merge nearby
    if len(spike_idx) > 1:
        merged_t, merged_groups = [spike_times[0]], [[]]
        for k in range(1, len(spike_idx)):
            if spike_idx[k] - spike_idx[k-1] <= MERGE_SAMPLES:
                pass
            else:
                merged_t.append(spike_times[k])
                merged_groups.append([])
        merged_t = np.array(merged_t)
    else:
        merged_t = spike_times

    intervals = np.diff(merged_t) if len(merged_t) > 1 else np.array([])

    print(f"\n  Combined ({SPIKE_LOGIC.upper()}) spike events: {len(merged_t)}")
    if len(intervals) > 0:
        cv = intervals.std() / intervals.mean() * 100
        regularity = ("very regular — hardware/driver timer"
                      if cv < 10 else
                      "moderate jitter"
                      if cv < 30 else
                      "irregular — external source")
        print(f"  Interval:  mean={intervals.mean():.2f} s  "
              f"std={intervals.std():.2f} s  "
              f"min={intervals.min():.2f} s  "
              f"max={intervals.max():.2f} s  "
              f"CV={cv:.1f}%")
        print(f"  → {regularity}")

    results["_meta"] = {
        "time": time, "dt": dt, "N": N, "cols": cols,
        "merged_t": merged_t, "intervals": intervals,
        "combined": combined,
    }
    return results

# ── Run on both files ─────────────────────────────────────────────────────────
raw_res  = group_spike_analysis(raw_df  := pd.read_csv(raw_path,  sep="\t"), "RAW FILE")
proc_res = group_spike_analysis(proc_df := pd.read_csv(proc_path, sep="\t"), "PROCESSED FILE")

# ── Timing alignment check ────────────────────────────────────────────────────
rt = raw_res["_meta"]["merged_t"]
pt = proc_res["_meta"]["merged_t"]
if len(rt) > 0 and len(pt) > 0:
    tol = raw_res["_meta"]["dt"] * 2
    matched = sum(1 for t in rt if np.any(np.abs(pt - t) <= tol))
    pct = matched / max(len(rt), len(pt)) * 100
    print(f"\n  Timing match raw↔processed: {matched}/{max(len(rt),len(pt))} ({pct:.0f}%)")
    print(f"  → {'Spikes in raw hardware data — not a processing artifact' if pct > 80 else 'Mismatch — investigate further'}")

# ── Plots ─────────────────────────────────────────────────────────────────────
GROUP_COLORS = {"DCDT": "steelblue", "Pressure": "darkorange", "Strain": "tomato"}
FILE_COLORS  = {"RAW FILE": "steelblue", "PROCESSED FILE": "darkorange"}

for res, df_used, file_label in [
    (raw_res,  raw_df,  "RAW FILE"),
    (proc_res, proc_df, "PROCESSED FILE"),
]:
    meta     = res["_meta"]
    time     = meta["time"]
    merged_t = meta["merged_t"]
    intervals= meta["intervals"]
    fcol     = FILE_COLORS[file_label]

    active_groups = [g for g in ["DCDT", "Pressure", "Strain"] if g in res]
    n_groups = len(active_groups)

    fig = plt.figure(figsize=(15, 4 + 2.5 * n_groups))
    fig.suptitle(f"Spike Analysis — {file_label}\n{os.path.basename(raw_path if 'RAW' in file_label else proc_path)}",
                 fontsize=11, fontweight="bold")

    gs = gridspec.GridSpec(n_groups + 2, 2, figure=fig, hspace=0.55, wspace=0.35)

    # Per-group spike score
    for row, gname in enumerate(active_groups):
        gr   = res[gname]
        gcol = GROUP_COLORS.get(gname, "gray")
        thresh = gr["thresh_count"]
        n_active = gr["active"].sum()

        ax = fig.add_subplot(gs[row, :])
        ax.plot(time[1:], gr["n_spiking"], color=gcol, lw=0.5, alpha=0.8)
        ax.axhline(thresh, color="red", lw=1, ls="--",
                   label=f"Threshold ({thresh}/{n_active} active ch)")
        for t in merged_t:
            ax.axvline(t, color="red", lw=0.7, alpha=0.3)
        ax.set_ylabel("# ch spiking")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{gname} ({len(gr['cols'])} ch total, {n_active} active)")
        ax.legend(fontsize=7, loc="upper right")

    # Interval vs time
    ax_iv = fig.add_subplot(gs[n_groups, 0])
    if len(intervals) > 0:
        ax_iv.plot(merged_t[:-1], intervals, "o-", color=fcol, ms=3)
        ax_iv.axhline(intervals.mean(), color="red", ls="--", lw=1,
                      label=f"Mean {intervals.mean():.2f} s")
        ax_iv.set_xlabel("Time (s)")
        ax_iv.set_ylabel("Interval (s)")
        ax_iv.set_title(f"Interval between events ({SPIKE_LOGIC.upper()} logic)")
        ax_iv.legend(fontsize=7)
    else:
        ax_iv.text(0.5, 0.5, "< 2 events", ha="center", va="center", transform=ax_iv.transAxes)
        ax_iv.set_title("Interval between events")

    # Interval histogram
    ax_hist = fig.add_subplot(gs[n_groups, 1])
    if len(intervals) > 0:
        ax_hist.hist(intervals, bins=min(30, max(5, len(intervals) // 2)),
                     color=fcol, edgecolor="white")
        ax_hist.axvline(intervals.mean(), color="red", ls="--", lw=1.5)
        ax_hist.set_xlabel("Interval (s)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Interval histogram\n(narrow = regular hardware period)")

    # Per-group channel hit rates side by side
    ax_ch = fig.add_subplot(gs[n_groups + 1, :])
    x_pos, x_labels, bar_colors = [], [], []
    offset = 0
    group_boundaries = []
    for gname in active_groups:
        gr   = res[gname]
        gcol = GROUP_COLORS.get(gname, "gray")
        n    = len(gr["cols"])
        positions = list(range(offset, offset + n))
        ax_ch.bar(positions, gr["hit_rate"], color=gcol, alpha=0.85, label=gname)
        x_pos   += positions
        x_labels += gr["cols"]
        offset  += n + 1
        group_boundaries.append(offset - 0.5)

    for b in group_boundaries[:-1]:
        ax_ch.axvline(b, color="gray", lw=1, ls=":")
    ax_ch.set_xticks(x_pos)
    ax_ch.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
    ax_ch.set_ylabel("% of events")
    ax_ch.set_ylim(0, 110)
    ax_ch.set_title("Channel hit rate by group  (inactive channels will show 0%)")
    ax_ch.legend(fontsize=8)

    fname = f"spike_analysis_{'raw' if 'RAW' in file_label else 'processed'}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {fname}")
    plt.show()
