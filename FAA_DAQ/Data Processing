import nidaqmx
from nidaqmx.constants import (BridgeConfiguration,
                                ExcitationSource,
                                StrainGageBridgeType)
from datetime import datetime
import matplotlib.pyplot as plt

# ─── CONFIGURATION ────────────────────────────────────────────
VOLTAGE_MODULE = "cDAQ1Mod1"  # Slot 1: NI-9205 (displacement + pressure)
STRAIN_MODULE  = "cDAQ1Mod2"  # Slot 2: NI-9235 (strain gauges)
SAMPLE_RATE    = 10           # Samples per second
SAMPLES_PER_READ = 10         # How many samples to read each loop

# ─── PRESSURE CONVERSION FORMULAS (one per channel) ──────────
def process_soil_plate_pressure(raw):
    return -2.86e-5 * raw**2 + 1.0038 * raw + 0.9331

def process_agg_plate_pressure(raw):
    return -1.34e-4 * raw**2 + 2.5171 * raw - 1.3375

def process_soil_pore_water_pressure(raw):
    return -2.79e-5 * raw**2 + 1.0006 * raw + 0.4014

def process_agg_pore_water_pressure(raw):
    return -2.93e-5 * raw**2 + 1.0073 * raw + 0.9260

# ─── MAIN ACQUISITION LOOP ────────────────────────────────────
def run_acquisition():
    with nidaqmx.Task() as strain_task, \
         nidaqmx.Task() as disp_task, \
         nidaqmx.Task() as pressure_task:

        # -- Strain channels (NI-9235, Slot 2, 8 channels) --
        for ch in range(8):
            strain_task.ai_channels.add_ai_strain_gage_chan(
                f"{STRAIN_MODULE}/ai{ch}",
                gage_factor=2.0,              # Update with your gauge factor
                initial_bridge_voltage=0.0,
                strain_config=StrainGageBridgeType.QUARTER_BRIDGE_I,
                voltage_excit_source=ExcitationSource.INTERNAL,
                voltage_excit_val=2.5,
                nominal_gage_resistance=120.0  # 120 ohm as per your hardware
            )

        # -- Displacement channels (NI-9205, Slot 1, ch0 to ch11) --
        for ch in range(12):
            disp_task.ai_channels.add_ai_voltage_chan(
                f"{VOLTAGE_MODULE}/ai{ch}",
                min_val=-10.0,
                max_val=10.0
            )

        # -- Pressure channels (NI-9205, Slot 1, ch17 to ch20) --
        for ch in [17, 18, 19, 20]:
            pressure_task.ai_channels.add_ai_voltage_chan(
                f"{VOLTAGE_MODULE}/ai{ch}",
                min_val=-10.0,
                max_val=10.0
            )

        # -- Set sample rates --
        strain_task.timing.cfg_samp_clk_timing(SAMPLE_RATE)
        disp_task.timing.cfg_samp_clk_timing(SAMPLE_RATE)
        pressure_task.timing.cfg_samp_clk_timing(SAMPLE_RATE)

        print("Acquisition started... Press Ctrl+C to stop")

        # Tare baselines (set on first read)
        baseline_strain = None
        baseline_disp   = None
        baseline_volt   = None   # [ch17, ch18, ch19, ch20] processed

        # Open raw and processed text files, write headers
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename  = f"data_raw_{ts}.txt"
        proc_filename = f"data_processed_{ts}.txt"
        raw_file  = open(raw_filename,  'w')
        proc_file = open(proc_filename, 'w')

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
            ["time"] +
            strain_names +
            disp_names +
            ["volt_ch17", "volt_ch18", "volt_ch19", "volt_ch20"]
        )
        proc_header = (
            ["time"] +
            strain_names +
            disp_names +
            ["soil_plate_pressure", "agg_plate_pressure",
             "soil_pore_water_pressure", "agg_pore_water_pressure"]
        )
        raw_file.write("\t".join(raw_header)   + "\n")
        proc_file.write("\t".join(proc_header) + "\n")

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

        # Plot 1: Time vs Strain
        ax1.set_title("Strain (tared)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Strain")
        strain_lines = [ax1.plot([], [], label=strain_names[i])[0] for i in range(8)]
        ax1.legend(fontsize=6, loc="upper left")

        # Plot 2: Time vs Displacement
        ax2.set_title("Displacement (tared)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement (V)")
        disp_lines = [ax2.plot([], [], label=disp_names[i])[0] for i in range(12)]
        ax2.legend(fontsize=6, loc="upper left")

        # Plot 3: Time vs Pressure
        press_labels = ["Soil Plate", "Agg Plate", "Soil Pore Water", "Agg Pore Water"]
        ax3.set_title("Pressure (tared)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure")
        press_lines = [ax3.plot([], [], label=press_labels[i])[0] for i in range(4)]
        ax3.legend(fontsize=7, loc="upper left")

        # Plot 4: Soil & Agg Plate Pressure vs DCDT_Left_Slab_B2_Bot
        ax4.set_title("Pressure vs DCDT_Left_Slab_B2_Bot")
        ax4.set_xlabel("DCDT_Left_Slab_B2_Bot (V)")
        ax4.set_ylabel("Pressure")
        soil_plate_line, = ax4.plot([], [], label="Soil Plate Pressure")
        agg_plate_line,  = ax4.plot([], [], label="Agg Plate Pressure")
        ax4.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        start_time = None

        try:
            while True:
                # Read all channels
                strain_data = strain_task.read(
                    number_of_samples_per_channel=SAMPLES_PER_READ)
                disp_data = disp_task.read(
                    number_of_samples_per_channel=SAMPLES_PER_READ)
                volt_data = pressure_task.read(
                    number_of_samples_per_channel=SAMPLES_PER_READ)

                # Per-channel values (first sample of each channel per row)
                strain_vals = [strain_data[i][0] for i in range(8)]
                disp_vals   = [disp_data[i][0]   for i in range(12)]

                # Process voltage channels (first sample of each channel)
                v17 = volt_data[0][0]
                v18 = volt_data[1][0]
                v19 = volt_data[2][0]
                v20 = volt_data[3][0]
                volt_vals = [process_soil_plate_pressure(v17),
                             process_agg_plate_pressure(v18),
                             process_soil_pore_water_pressure(v19),
                             process_agg_pore_water_pressure(v20)]

                # Set baseline on very first read
                if baseline_strain is None:
                    baseline_strain = strain_vals[:]
                    baseline_disp   = disp_vals[:]
                    baseline_volt   = volt_vals[:]

                # Apply tare: rowN - row1
                strain_tared = [strain_vals[i] - baseline_strain[i] for i in range(8)]
                disp_tared   = [disp_vals[i]   - baseline_disp[i]   for i in range(12)]
                volt_tared   = [volt_vals[i]    - baseline_volt[i]   for i in range(4)]

                # ── Update live plots ─────────────────────────────
                if start_time is None:
                    start_time = datetime.now()
                elapsed = (datetime.now() - start_time).total_seconds()
                t_data.append(elapsed)

                for i in range(8):
                    strain_plot[i].append(strain_tared[i])
                for i in range(12):
                    disp_plot[i].append(disp_tared[i])
                for i in range(4):
                    press_plot[i].append(volt_tared[i])
                b2bot_plot.append(disp_tared[6])   # DCDT_Left_Slab_B2_Bot

                for i, line in enumerate(strain_lines):
                    line.set_data(t_data, strain_plot[i])
                for i, line in enumerate(disp_lines):
                    line.set_data(t_data, disp_plot[i])
                for i, line in enumerate(press_lines):
                    line.set_data(t_data, press_plot[i])
                soil_plate_line.set_data(b2bot_plot, press_plot[0])
                agg_plate_line.set_data(b2bot_plot, press_plot[1])

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.relim()
                    ax.autoscale_view()
                plt.pause(0.001)

                # Print real-time summary
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print(f"\n[{timestamp}]")
                print(f"Strain tared:   {[round(v, 6) for v in strain_tared]}")
                print(f"Disp tared:     {[round(v, 4) for v in disp_tared]}")
                print(f"Pressure tared - SoilPlate:{round(volt_tared[0],4)}  AggPlate:{round(volt_tared[1],4)}  SoilPore:{round(volt_tared[2],4)}  AggPore:{round(volt_tared[3],4)}")

                # Write raw data row (raw voltage, not converted or tared)
                raw_row = [timestamp] + \
                          [f"{v:.6f}" for v in strain_vals] + \
                          [f"{v:.6f}" for v in disp_vals] + \
                          [f"{v17:.6f}", f"{v18:.6f}", f"{v19:.6f}", f"{v20:.6f}"]
                raw_file.write("\t".join(raw_row) + "\n")
                raw_file.flush()

                # Write processed and tared data row
                proc_row = [timestamp] + \
                           [f"{v:.6f}" for v in strain_tared] + \
                           [f"{v:.6f}" for v in disp_tared] + \
                           [f"{v:.6f}" for v in volt_tared]
                proc_file.write("\t".join(proc_row) + "\n")
                proc_file.flush()

        except KeyboardInterrupt:
            print("\nAcquisition stopped.")
            raw_file.close()
            proc_file.close()
            print(f"Raw data saved to      {raw_filename}")
            print(f"Processed data saved to {proc_filename}")

# ─── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    run_acquisition()