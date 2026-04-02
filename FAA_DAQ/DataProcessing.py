import nidaqmx
from nidaqmx.constants import (BridgeConfiguration,
                                ExcitationSource,
                                StrainGageBridgeType,
                                TerminalConfiguration)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import os

# ─── CONFIGURATION ────────────────────────────────────────────
VOLTAGE_MODULE = "cDAQ1Mod1"  # Slot 1: NI-9205 (displacement + pressure)
STRAIN_MODULE  = "cDAQ1Mod2"  # Slot 2: NI-9235 (strain gauges)
SAMPLE_RATE      = 16         # Voltage/pressure effective sample rate (Hz)
SAMPLES_PER_READ = 1          # Voltage samples per loop iteration
STRAIN_HW_RATE   = 974        # NI-9235 minimum hardware sample rate (Hz)
STRAIN_DOWNSAMPLE = round(STRAIN_HW_RATE / SAMPLE_RATE)  # 974/16 ≈ 61 — strain samples to average per voltage sample
RECORD_DELAY     = 10         # Seconds to wait before writing to file (time to start MTS)
TARE_SAMPLES     = 16        # Samples to average for tare baseline (160 = 10 seconds at 16 Hz)
DISP_SCALE       = 3.937 / 10.0  # V → inches (0 V = 0 in, 10 V = 3.937 in)
KPA_TO_KSI       = 0.000145038   # kPa → ksi conversion

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

        # -- Set sample rates --
        # NI-9235 cannot go below 974 Hz — run at hardware minimum and downsample in software
        strain_task.timing.cfg_samp_clk_timing(STRAIN_HW_RATE)
        voltage_task.timing.cfg_samp_clk_timing(SAMPLE_RATE)

        print("Acquisition started... Press Ctrl+C to stop")
        print(f"Waiting {RECORD_DELAY}s before tare — start MTS machine now...")

        # Baselines set after delay + tare collection
        baseline_strain    = None
        baseline_disp_in   = None
        baseline_press_kpa = None

        # Tare accumulators
        tare_strain    = [0.0] * 8
        tare_disp_in   = [0.0] * 12
        tare_press_kpa = [0.0] * 4
        tare_collected = 0

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
            ["soil_plate_pressure_ksi", "agg_plate_pressure_ksi",
             "soil_pore_water_pressure_ksi", "agg_pore_water_pressure_ksi"] +
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

        # Data storage lists
        t_data          = []
        strain_plot     = [[] for _ in range(8)]
        disp_plot       = [[] for _ in range(12)]
        press_plot      = [[] for _ in range(4)]
        b2bot_plot      = []   # DCDT_Left_Slab_B2_Bot (disp index 6)

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

        # Plot 3: Time vs Pressure (processed, ksi)
        press_labels = ["Soil Plate", "Agg Plate", "Soil Pore Water", "Agg Pore Water"]
        ax3.set_title("Pressure - Processed")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (ksi)")
        press_lines = [ax3.plot([], [], label=press_labels[i])[0] for i in range(4)]
        ax3.legend(fontsize=7, loc="upper left")

        # Plot 4: Pressure (ksi) vs Displacement (in)
        ax4.set_title("Pressure vs DCDT_Left_Slab_B2_Bot")
        ax4.set_xlabel("DCDT_Left_Slab_B2_Bot (in)")
        ax4.set_ylabel("Pressure (ksi)")
        soil_plate_line, = ax4.plot([], [], label="Soil Plate Pressure (ksi)")
        agg_plate_line,  = ax4.plot([], [], label="Agg Plate Pressure (ksi)")
        ax4.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plt.show(block=False)
        start_time   = None
        sample_count = 0   # counts written samples; time = sample_count / SAMPLE_RATE

        try:
            while True:
                # Read voltage channels (16 Hz) and strain channels (974 Hz, then average)
                # STRAIN_DOWNSAMPLE strain samples are collected per voltage sample
                # and averaged to produce one strain value at the effective 16 Hz rate
                strain_data  = strain_task.read(
                    number_of_samples_per_channel=STRAIN_DOWNSAMPLE)
                voltage_data = voltage_task.read(
                    number_of_samples_per_channel=SAMPLES_PER_READ)

                if start_time is None:
                    start_time = datetime.now()

                elapsed_total = (datetime.now() - start_time).total_seconds()

                # ── Loop over voltage samples (1 per iteration at 16 Hz) ──
                n_samples = len(voltage_data[0])
                for s in range(n_samples):
                    # ── Step 1: Read raw values from hardware ─────
                    # Strain: average all STRAIN_DOWNSAMPLE hardware samples → 1 value at 16 Hz
                    strain_raw = [sum(strain_data[i]) / STRAIN_DOWNSAMPLE for i in range(8)]
                    disp_raw   = [voltage_data[i][s] for i in range(12)]
                    v17 = voltage_data[12][s]
                    v18 = voltage_data[13][s]
                    v19 = voltage_data[14][s]
                    v20 = voltage_data[15][s]

                    # ── Step 2: Convert raw to physical units ─────
                    disp_in   = [v * DISP_SCALE for v in disp_raw]
                    press_kpa = [process_soil_plate_pressure(v17),
                                 process_agg_plate_pressure(v18),
                                 process_soil_pore_water_pressure(v19),
                                 process_agg_pore_water_pressure(v20)]

                    # ── Phase A: still in delay — just plot, no tare, no file ─
                    if elapsed_total < RECORD_DELAY:
                        remaining = RECORD_DELAY - elapsed_total
                        if s == 0:
                            print(f"\r[WAITING — recording starts in {remaining:.1f}s]", end="")
                        continue   # skip tare collection and file writing

                    # ── Phase B: collecting tare samples after delay ───────
                    if tare_collected < TARE_SAMPLES:
                        for i in range(8):
                            tare_strain[i]    += strain_raw[i]
                        for i in range(12):
                            tare_disp_in[i]   += disp_in[i]
                        tare_press_kpa[0] += press_kpa[0]
                        tare_press_kpa[1] += press_kpa[1]
                        tare_press_kpa[2] += press_kpa[2]
                        tare_press_kpa[3] += press_kpa[3]
                        tare_collected += 1
                        if tare_collected == TARE_SAMPLES:
                            baseline_strain    = [round(v / TARE_SAMPLES, 6) for v in tare_strain]
                            baseline_disp_in   = [round(v / TARE_SAMPLES, 6) for v in tare_disp_in]
                            baseline_press_kpa = [round(v / TARE_SAMPLES, 6) for v in tare_press_kpa]
                            print(f"\nTare complete ({TARE_SAMPLES} samples). Recording started.")
                        continue   # skip file writing until baseline is ready

                    # ── Phase C: baseline ready — tare, write, plot ────────
                    strain_tared    = [strain_raw[i] - baseline_strain[i]    for i in range(8)]
                    disp_tared      = [disp_in[i]    - baseline_disp_in[i]   for i in range(12)]
                    press_tared_kpa = [press_kpa[i]  - baseline_press_kpa[i] for i in range(4)]
                    press_tared_ksi = [v * KPA_TO_KSI for v in press_tared_kpa]

                    t = sample_count / SAMPLE_RATE
                    sample_count += 1

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
                               [f"{v:.6f}" for v in press_tared_ksi] + \
                               [f"{v:.6f}" for v in strain_tared]
                    proc_file.write("\t".join(proc_row) + "\n")

                    t_data.append(t)
                    for i in range(8):
                        strain_plot[i].append(strain_tared[i])
                    for i in range(12):
                        disp_plot[i].append(disp_tared[i])
                    for i in range(4):
                        press_plot[i].append(press_tared_ksi[i])
                    b2bot_plot.append(disp_tared[6])

                raw_file.flush()
                proc_file.flush()
                cal_file.flush()

                # ── Refresh plots once per batch ──────────────────
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