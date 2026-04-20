import nidaqmx
from nidaqmx.constants import (BridgeConfiguration,
                                ExcitationSource,
                                StrainGageBridgeType,
                                TerminalConfiguration)
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import os
from collections import deque

# ─── CONFIGURATION ────────────────────────────────────────────
VOLTAGE_MODULE = "cDAQ1Mod1"  # Slot 1: NI-9205 (displacement + pressure)
STRAIN_MODULE  = "cDAQ1Mod2"  # Slot 2: NI-9235 (strain gauges)
SAMPLE_RATE      = 16         # Effective output rate (Hz) — written to file and plotted
HW_RATE          = 794        # Hardware clock rate for both tasks (NI-9235 minimum)
DOWNSAMPLE       = round(HW_RATE / SAMPLE_RATE)  # 794/16 ≈ 50 — samples averaged per output sample
WALK_COUNTDOWN   = 10         # Seconds countdown before recording — time to walk to MTS
RAMP_SAMPLES     = 160        # Samples during ramp (10s × 16Hz) — averaged for 1 kip baseline
DISP_SCALE       = 3.937 / 10.0  # V → inches (0 V = 0 in, 10 V = 3.937 in)
KPA_TO_PSI       = 0.145038      # kPa → psi conversion

# ─── PRESSURE CONVERSION FORMULAS (one per channel) ──────────
def process_soil_plate_pressure(raw):
    return -2.86e-5 * raw**2 + 1.0038 * raw + 0.9331

def process_agg_plate_pressure(raw):
    return -1.34e-4 * raw**2 + 2.5171 * raw - 1.3375

def process_soil_pore_water_pressure(raw):
    return -2.79e-5 * raw**2 + 1.0006 * raw + 0.4014

def process_agg_pore_water_pressure(raw):
    return -2.93e-5 * raw**2 + 1.0073 * raw + 0.9260

SMOOTH_WINDOW = 11   # increase for smoother curves

def smooth(y):
    if len(y) < SMOOTH_WINDOW:
        return y
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return np.convolve(y, kernel, mode='same')

# ─── MAIN ACQUISITION LOOP ────────────────────────────────────
def run_acquisition():
    with nidaqmx.Task() as strain_task, \
         nidaqmx.Task() as voltage_task:

        # -- Strain channels (NI-9235, Slot 2, 8 channels) --
        for ch in range(8):
            strain_task.ai_channels.add_ai_strain_gage_chan(
                f"{STRAIN_MODULE}/ai{ch}",
                gage_factor=2.0,              # Update with your gauge factor
                initial_bridge_voltage=0.0,
                strain_config=StrainGageBridgeType.QUARTER_BRIDGE_I,
                voltage_excit_source=ExcitationSource.INTERNAL,
                voltage_excit_val=2.0,
                nominal_gage_resistance=120.0  # 120 ohm as per your hardware
            )

        # -- Displacement + Pressure channels combined in one task (NI-9205, Slot 1) --
        # Indices 0-11  → displacement (ai0 to ai11)
        # Indices 12-15 → pressure     (ai17 to ai20)
        for ch in range(12):
            voltage_task.ai_channels.add_ai_voltage_chan(
                f"{VOLTAGE_MODULE}/ai{ch}",
                terminal_config=TerminalConfiguration.RSE,
                min_val=0,
                max_val=10.0
            )
        for ch in [17, 18, 19, 20]:
            voltage_task.ai_channels.add_ai_voltage_chan(
                f"{VOLTAGE_MODULE}/ai{ch}",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=0,
                max_val=10.0
            )

        # -- Set sample rates — large buffer (10s) to absorb plotting/IO delays --
        # NI-9235 does not expose SampleClock; both tasks run independently
        strain_task.timing.cfg_samp_clk_timing(HW_RATE, samps_per_chan=HW_RATE * 10)
        voltage_task.timing.cfg_samp_clk_timing(HW_RATE, samps_per_chan=HW_RATE * 10)

        # ── 10s countdown — walk to MTS during this time ─────────
        print("Acquisition started. Walk to MTS now...")
        for i in range(WALK_COUNTDOWN, 0, -1):
            print(f"\r  Starting in {i}s... ", end="", flush=True)
            time.sleep(1)
        print("\rRecording started — start MTS ramp now!          ")

        # Baselines set from ramp data — None until ramp completes
        baseline_strain    = None
        baseline_disp_in   = None
        baseline_press_kpa = None

        # Sample counter for drift-free timestamps starting at t=0
        output_sample_count = 0

        # Ramp tare accumulators
        ramp_strain    = [0.0] * 8
        ramp_disp_in   = [0.0] * 12
        ramp_press_kpa = [0.0] * 4
        ramp_collected = 0

        # Buffer for processed rows during ramp (can't tare until ramp ends)
        proc_buffer = []
        plot_counter = 0

        # Auto-zero detection: skip samples where any displacement channel
        # jumps more than this threshold in one sample (NI-9235 calibration artifact)
        AUTOZERO_THRESHOLD = 0.3   # volts — any jump larger than this is flagged
        prev_disp_raw = None       # updated only on accepted samples

        # Open raw, processed, and calibrated text files, write headers
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename  = f"data_raw_{ts}.txt"
        proc_filename = f"data_processed_{ts}.txt"
        cal_filename  = f"data_calibrated_{ts}.txt"
        raw_file  = open(raw_filename,  'w')
        proc_file = open(proc_filename, 'w')
        cal_file  = open(cal_filename,  'w')

        strain_names = [
            "SG_2E_top", "SG_3E_top", "SG_4E_top", "SG_4E_bot",
            "SG_5E_top", "SG_5E_bot", "SG_6E_top", "SG_7E_top"
        ]
        disp_names = [
            "DCDT_Right_Slab_A1", "DCDT_Right_Slab_A2", "DCDT_Right_Slab_A3",
            "DCDT_Right_Slab_B1", "DCDT_Right_Slab_B3",
            "DCDT_Left_Slab_B1",  "DCDT_Left_Slab_B2_Bot", "DCDT_Left_Slab_B3",
            "DCDT_Left_Slab_C1",  "DCDT_Left_Slab_C2", "DCDT_Left_Slab_C3",
            "DCDT_Beam_B2_Top"
        ]

        raw_header = (
            ["time_s"] +
            disp_names +
            ["volt_ch17", "volt_ch18", "volt_ch19", "volt_ch20"] +
            strain_names
        )
        proc_header = (
            ["time_s"] +
            disp_names +
            ["soil_plate_pressure_psi", "agg_plate_pressure_psi",
             "soil_pore_water_pressure_psi", "agg_pore_water_pressure_psi"] +
            strain_names
        )
        cal_header = (
            ["time_s"] +
            disp_names +
            ["soil_plate_pressure_kPa", "agg_plate_pressure_kPa",
             "soil_pore_water_pressure_kPa", "agg_pore_water_pressure_kPa"]
        )
        raw_file.write("\t".join(raw_header)   + "\n")
        proc_file.write("\t".join(proc_header) + "\n")
        cal_file.write("\t".join(cal_header)   + "\n")

        # ── Live plot setup ───────────────────────────────────────
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Live DAQ Data")

        # Rolling window for live plots — last 1 minute of data
        PLOT_WINDOW = 60 * SAMPLE_RATE     # 960 points at 16 Hz
        t_data      = deque(maxlen=PLOT_WINDOW)
        strain_plot = [deque(maxlen=PLOT_WINDOW) for _ in range(8)]
        disp_plot   = [deque(maxlen=PLOT_WINDOW) for _ in range(12)]
        press_plot  = [deque(maxlen=PLOT_WINDOW) for _ in range(4)]
        b2bot_plot  = deque(maxlen=PLOT_WINDOW)

        # Full dataset retained in memory for end-of-test plot saves
        t_full      = []
        strain_full = [[] for _ in range(8)]
        disp_full   = [[] for _ in range(12)]
        press_full  = [[] for _ in range(4)]
        b2bot_full  = []

        # Plot 1: Time vs Strain (tared)
        ax1.set_title("Strain (tared)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Strain (microstrain)")
        strain_lines = [ax1.plot([], [], label=strain_names[i])[0] for i in range(8)]
        ax1.legend(fontsize=6, loc="upper left")

        # Plot 2: Time vs Displacement (processed, in)
        ax2.set_title("Displacement - Processed")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement (in)")
        disp_lines = [ax2.plot([], [], label=disp_names[i])[0] for i in range(12)]
        ax2.legend(fontsize=6, loc="upper left")

        # Plot 3: Time vs Pressure (processed, psi)
        press_labels = ["Soil Plate", "Agg Plate", "Soil Pore Water", "Agg Pore Water"]
        ax3.set_title("Pressure - Processed")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (psi)")
        press_lines = [ax3.plot([], [], label=press_labels[i])[0] for i in range(4)]
        ax3.legend(fontsize=7, loc="upper left")

        # Plot 4: Pressure (psi) vs Displacement (in)
        ax4.set_title("Pressure vs DCDT_Left_Slab_B2_Bot")
        ax4.set_xlabel("DCDT_Left_Slab_B2_Bot (in)")
        ax4.set_ylabel("Pressure (psi)")
        soil_plate_line, = ax4.plot([], [], label="Soil Plate Pressure (psi)")
        agg_plate_line,  = ax4.plot([], [], label="Agg Plate Pressure (psi)")
        ax4.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plt.show(block=False)
        try:
            while True:
                # Read voltage channels (16 Hz) and strain channels (974 Hz, then average)
                # STRAIN_DOWNSAMPLE strain samples are collected per voltage sample
                # and averaged to produce one strain value at the effective 16 Hz rate
                strain_data  = strain_task.read(
                    number_of_samples_per_channel=DOWNSAMPLE)
                voltage_data = voltage_task.read(
                    number_of_samples_per_channel=DOWNSAMPLE)


                # Both tasks read DOWNSAMPLE=61 samples per loop at 974 Hz (shared clock)
                # Average all 61 → 1 output value per channel at effective 16 Hz
                strain_raw = [sum(strain_data[i])  / DOWNSAMPLE for i in range(8)]
                disp_raw   = [sum(voltage_data[i]) / DOWNSAMPLE for i in range(12)]
                v17 = sum(voltage_data[12]) / DOWNSAMPLE
                v18 = sum(voltage_data[13]) / DOWNSAMPLE
                v19 = sum(voltage_data[14]) / DOWNSAMPLE
                v20 = sum(voltage_data[15]) / DOWNSAMPLE

                # ── Auto-zero detection: skip NI-9235 calibration glitches ──
                if prev_disp_raw is not None:
                    if any(abs(disp_raw[i] - prev_disp_raw[i]) > AUTOZERO_THRESHOLD
                           for i in range(12)):
                        continue   # discard this sample entirely
                prev_disp_raw = disp_raw

                # ── Step 2: Convert raw to physical units ─────────
                disp_in   = [v * DISP_SCALE for v in disp_raw]
                # Pressure formulas expect millivolts — convert from volts first
                press_kpa = [process_soil_plate_pressure(v17 * 1000),
                             process_agg_plate_pressure(v18 * 1000),
                             process_soil_pore_water_pressure(v19 * 1000),
                             process_agg_pore_water_pressure(v20 * 1000)]

                # ── Phase A: collect ramp samples for baseline ────
                if ramp_collected < RAMP_SAMPLES:
                    for i in range(8):
                        ramp_strain[i]    += strain_raw[i]
                    for i in range(12):
                        ramp_disp_in[i]   += disp_in[i]
                    ramp_press_kpa[0] += press_kpa[0]
                    ramp_press_kpa[1] += press_kpa[1]
                    ramp_press_kpa[2] += press_kpa[2]
                    ramp_press_kpa[3] += press_kpa[3]
                    ramp_collected += 1

                    # Write raw and calibrated immediately
                    output_sample_count += 1
                    t = output_sample_count / SAMPLE_RATE
                    raw_row = [f"{t:.6f}"] + \
                              [f"{v:.6f}" for v in disp_raw] + \
                              [f"{v17:.6f}", f"{v18:.6f}", f"{v19:.6f}", f"{v20:.6f}"] + \
                              [f"{v:.6f}" for v in strain_raw]
                    raw_file.write("\t".join(raw_row) + "\n")
                    cal_row = [f"{t:.6f}"] + \
                              [f"{v:.6f}" for v in disp_in] + \
                              [f"{v:.6f}" for v in press_kpa]
                    cal_file.write("\t".join(cal_row) + "\n")

                    # Buffer processed row (no baseline yet)
                    proc_buffer.append((t, disp_in[:], press_kpa[:], strain_raw[:]))

                    # Once ramp complete → lock baseline, flush buffered processed rows
                    if ramp_collected == RAMP_SAMPLES:
                        baseline_strain    = [round(v / RAMP_SAMPLES, 6) for v in ramp_strain]
                        baseline_disp_in   = [round(v / RAMP_SAMPLES, 6) for v in ramp_disp_in]
                        baseline_press_kpa = [round(v / RAMP_SAMPLES, 6) for v in ramp_press_kpa]
                        print(f"\nRamp complete. Baseline set at 1 kip. Cyclic recording started.")
                        for (bt, bd, bp, bs) in proc_buffer:
                            d_tared = [bd[i] - baseline_disp_in[i]   for i in range(12)]
                            p_tared = [(bp[i] - baseline_press_kpa[i]) * KPA_TO_PSI for i in range(4)]
                            s_tared = [bs[i] - baseline_strain[i]    for i in range(8)]
                            proc_row = [f"{bt:.6f}"] + \
                                       [f"{v:.6f}" for v in d_tared] + \
                                       [f"{v:.6f}" for v in p_tared] + \
                                       [f"{v:.6f}" for v in s_tared]
                            proc_file.write("\t".join(proc_row) + "\n")
                        proc_buffer.clear()
                    continue

                # ── Phase B: cyclic recording — baseline ready ────
                strain_tared    = [strain_raw[i] - baseline_strain[i]    for i in range(8)]
                disp_tared      = [disp_in[i]    - baseline_disp_in[i]   for i in range(12)]
                press_tared_kpa = [press_kpa[i]  - baseline_press_kpa[i] for i in range(4)]
                press_tared_psi = [v * KPA_TO_PSI for v in press_tared_kpa]

                output_sample_count += 1
                t = output_sample_count / SAMPLE_RATE

                raw_row = [f"{t:.6f}"] + \
                          [f"{v:.6f}" for v in disp_raw] + \
                          [f"{v17:.6f}", f"{v18:.6f}", f"{v19:.6f}", f"{v20:.6f}"] + \
                          [f"{v:.6f}" for v in strain_raw]
                raw_file.write("\t".join(raw_row) + "\n")

                cal_row = [f"{t:.6f}"] + \
                          [f"{v:.6f}" for v in disp_in] + \
                          [f"{v:.6f}" for v in press_kpa]
                cal_file.write("\t".join(cal_row) + "\n")

                proc_row = [f"{t:.6f}"] + \
                           [f"{v:.6f}" for v in disp_tared] + \
                           [f"{v:.6f}" for v in press_tared_psi] + \
                           [f"{v:.6f}" for v in strain_tared]
                proc_file.write("\t".join(proc_row) + "\n")

                t_data.append(t)
                t_full.append(t)
                for i in range(8):
                    strain_plot[i].append(strain_tared[i])
                    strain_full[i].append(strain_tared[i])
                for i in range(12):
                    disp_plot[i].append(disp_tared[i])
                    disp_full[i].append(disp_tared[i])
                for i in range(4):
                    press_plot[i].append(press_tared_psi[i])
                    press_full[i].append(press_tared_psi[i])
                b2bot_plot.append(disp_tared[6])
                b2bot_full.append(disp_tared[6])

                raw_file.flush()
                proc_file.flush()
                cal_file.flush()

                # ── Refresh plots once per second (every SAMPLE_RATE loops) ──
                plot_counter += 1
                if plot_counter >= SAMPLE_RATE:
                    plot_counter = 0
                    for i, line in enumerate(strain_lines):
                        line.set_data(t_data, smooth(strain_plot[i]))
                    for i, line in enumerate(disp_lines):
                        line.set_data(t_data, smooth(disp_plot[i]))
                    for i, line in enumerate(press_lines):
                        line.set_data(t_data, smooth(press_plot[i]))
                    soil_plate_line.set_data(smooth(b2bot_plot), smooth(press_plot[0]))
                    agg_plate_line.set_data(smooth(b2bot_plot), smooth(press_plot[1]))

                    for ax in [ax1, ax2, ax3, ax4]:
                        ax.relim()
                        ax.autoscale_view()
                    plt.pause(0.001)


        except KeyboardInterrupt:
            print("\nAcquisition stopped.")
            raw_file.close()
            proc_file.close()
            cal_file.close()

            # ── Popup: ask whether to save data files ─────────────
            root = tk.Tk()
            root.withdraw()  # hide the root window

            save_data = messagebox.askyesno(
                "Save Data Files",
                "Do you want to save the data files?\n\n"
                f"  • {raw_filename}\n"
                f"  • {proc_filename}\n"
                f"  • {cal_filename}"
            )
            if save_data:
                print(f"Raw data saved to        {raw_filename}")
                print(f"Processed data saved to  {proc_filename}")
                print(f"Calibrated data saved to {cal_filename}")
            else:
                os.remove(raw_filename)
                os.remove(proc_filename)
                os.remove(cal_filename)
                print("Data files discarded.")

            # ── Popup: ask whether to save plots ──────────────────
            save_plots = messagebox.askyesno(
                "Save Plots",
                "Do you want to save the live plots as PNG images?"
            )
            if save_plots:
                # Redraw all axes with full dataset before saving
                for i, line in enumerate(strain_lines):
                    line.set_data(t_full, smooth(strain_full[i]))
                for i, line in enumerate(disp_lines):
                    line.set_data(t_full, smooth(disp_full[i]))
                for i, line in enumerate(press_lines):
                    line.set_data(t_full, smooth(press_full[i]))
                soil_plate_line.set_data(smooth(b2bot_full), smooth(press_full[0]))
                agg_plate_line.set_data(smooth(b2bot_full), smooth(press_full[1]))
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.relim()
                    ax.autoscale_view()
                fig.canvas.draw()

                plot_info = [
                    (ax1, "plot_strain"),
                    (ax2, "plot_displacement"),
                    (ax3, "plot_pressure"),
                    (ax4, "plot_pressure_vs_disp"),
                ]
                for ax, name in plot_info:
                    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                        fig.dpi_scale_trans.inverted())
                    fig_name = f"{name}_{ts}.png"
                    fig.savefig(fig_name, bbox_inches=extent)
                    print(f"Plot saved to            {fig_name}")
            else:
                print("Plots discarded.")

            root.destroy()

# ─── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    run_acquisition()