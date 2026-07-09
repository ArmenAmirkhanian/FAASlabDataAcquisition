"""
Consolidation Test — Dial Gauge Photo Capture
==============================================
Nikon D3300 + digiCamControl + Windows

Run this at the start of each load step. It will:
  1. Ask you which load step you're on
  2. Ask you to pick a save folder
  3. Count down 5 seconds so you can apply the load
  4. Fire the camera at every ASTM time interval automatically
  5. Beep before each shot so you know it's about to fire
  6. Save a log file alongside the photos

Requirements:
  - digiCamControl installed at default path
  - Camera connected via USB and turned ON
  - digiCamControl app open in background (keeps the USB driver alive)

Usage:
  python consolidation_capture.py
"""

import time
import os
import sys
import msvcrt
import urllib.request
import urllib.parse
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog, messagebox

# ── digiCamControl webserver ──────────────────────────────────────
# Talks to the already-running digiCamControl GUI over its built-in
# webserver instead of spawning a new CameraControlCmd.exe per shot.
# Spawning a fresh process per shot forces a full USB/PTP reconnect
# each time, which raced with the GUI's own open connection and
# caused capture calls to hang/time out mid-session.
WEBSERVER = "http://127.0.0.1:5513"

# ── ASTM D2435 time schedule ──────────────────────────────────────
# (label shown in filename, seconds from t=0)
SCHEDULE = [
    ("00_0sec",    0),
    ("01_6sec",    6),
    ("02_15sec",   15),
    ("03_30sec",   30),
    ("04_1min",    60),
    ("05_2min",    120),
    ("06_4min",    240),
    ("07_8min",    480),
    ("08_15min",   900),
    ("09_30min",   1800),
    ("10_1hr",     3600),
    ("11_2hr",     7200),
    ("12_4hr",     14400),
    ("13_8hr",     28800),
    ("14_24hr",    86400),
]

# Load step info — matches your Excel (kg values)
LOAD_STEPS = {
    1:  {"phase": "Loading",   "kg": 1.0},
    2:  {"phase": "Loading",   "kg": 2.0},
    3:  {"phase": "Loading",   "kg": 4.0},
    4:  {"phase": "Loading",   "kg": 8.0},
    5:  {"phase": "Loading",   "kg": 16.0},
    6:  {"phase": "Unloading", "kg": 8.0},
    7:  {"phase": "Unloading", "kg": 4.0},
    8:  {"phase": "Unloading", "kg": 2.0},
    9:  {"phase": "Reloading", "kg": 4.0},
    10: {"phase": "Reloading", "kg": 8.0},
    11: {"phase": "Reloading", "kg": 16.0},
    12: {"phase": "Loading",   "kg": 32.0},
    13: {"phase": "Loading",   "kg": 64.0},
}


def beep(n=1):
    """Windows terminal beep — audible cue before capture."""
    for _ in range(n):
        print("\a", end="", flush=True)
        time.sleep(0.15)


def web_get(params, timeout=10):
    """Send one command to digiCamControl's webserver and return the response text."""
    url = f"{WEBSERVER}/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore").strip()


def check_webserver():
    """Make sure digiCamControl's GUI is running with its webserver enabled."""
    try:
        web_get({"slc": "get", "param1": "lastcaptured", "param2": ""})
    except Exception:
        print("\nERROR: Can't reach digiCamControl's webserver at", WEBSERVER)
        print("  - Open digiCamControl (the GUI, not just installed)")
        print("  - Settings -> Webserver -> check 'Use web server' (port 5513)")
        print("  - Restart digiCamControl after enabling it")
        input("\nPress Enter to exit...")
        sys.exit(1)


def test_camera():
    """Fire a test capture to confirm camera is responding."""
    print("\nTesting camera connection...")
    try:
        response = web_get({"slc": "capture", "param1": "", "param2": ""})
    except Exception as e:
        print(f"\nERROR: Webserver request failed: {e}")
        return False

    if "error" in response.lower():
        print(f"\nERROR: Camera not responding: {response}")
        print("  - Check USB cable is plugged in")
        print("  - Make sure camera is ON")
        return False

    print("  Camera detected OK.")
    return True


def capture_photo(save_path, session_dir):
    """
    Trigger one capture via digiCamControl's webserver and move the
    resulting file to save_path. Reuses the GUI's already-open camera
    connection instead of spawning a new process per shot.
    """
    try:
        web_get({"slc": "capture", "param1": "", "param2": ""})
    except Exception as e:
        return False, f"capture request failed: {e}"

    # Poll for the filename digiCamControl assigned to this capture
    deadline = time.monotonic() + 15
    captured = "-"
    while time.monotonic() < deadline:
        try:
            captured = web_get({"slc": "get", "param1": "lastcaptured", "param2": ""})
        except Exception:
            captured = "-"
        if captured and captured != "-":
            break
        time.sleep(0.3)

    if not captured or captured == "-":
        return False, "timed out waiting for capture confirmation"

    src = os.path.join(session_dir, os.path.basename(captured))
    time.sleep(0.5)   # let the file finish writing to disk
    if not os.path.exists(src):
        return False, f"expected file not found: {src}"

    try:
        os.replace(src, save_path)
    except OSError as e:
        return False, f"could not rename {src}: {e}"

    return True, captured


def pick_folder():
    """Open a folder picker dialog."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(
        title="Choose folder to save consolidation photos"
    )
    root.destroy()
    return folder


def pick_load_step():
    """Ask which load step via simple dialog."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # Build a simple choice string
    options = "\n".join(
        f"  {k}: {v['phase']:10}  {v['kg']} kg"
        for k, v in LOAD_STEPS.items()
    )
    step = simpledialog.askinteger(
        "Load Step",
        f"Which load step are you starting?\n\n{options}\n\nEnter step number (1-13):",
        minvalue=1, maxvalue=13,
        parent=root
    )
    root.destroy()
    return step


def format_time(seconds):
    """Human-readable time remaining."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m"


def run_session(save_folder, step_num):
    """Main capture loop."""
    step_info  = LOAD_STEPS[step_num]
    phase      = step_info["phase"]
    kg         = step_info["kg"]
    session_id = f"Step{step_num:02d}_{phase}_{kg}kg"

    # Create session subfolder
    session_dir = os.path.join(save_folder, session_id)
    os.makedirs(session_dir, exist_ok=True)
    web_get({"slc": "set", "param1": "session.folder", "param2": session_dir})

    # Log file
    log_path = os.path.join(session_dir, "capture_log.txt")
    log = open(log_path, "w", encoding="utf-8")
    log.write("CONSOLIDATION TEST — CAPTURE LOG\n")
    log.write(f"Step:    {step_num}  ({phase}  {kg} kg)\n")
    log.write(f"Folder:  {session_dir}\n")
    log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log.write(f"{'#':<4} {'Label':<14} {'Target':>8} {'Actual':>8}  {'Status':<8}  File\n")
    log.write("-" * 75 + "\n")

    print(f"\n{'='*55}")
    print(f"  STEP {step_num}  |  {phase}  |  {kg} kg")
    print(f"  Saving to: {session_dir}")
    print(f"  Photos: {len(SCHEDULE)}")
    print(f"{'='*55}")
    print()
    print("  GET READY — apply the load now.")
    print()

    # Countdown before t=0
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...", end="\r")
        time.sleep(1)

    beep(2)
    print("\n  GO — t = 0                                    ")
    print()

    t_start = time.monotonic()

    for idx, (label, target_sec) in enumerate(SCHEDULE):
        # Wait until it's time
        elapsed = time.monotonic() - t_start
        wait    = target_sec - elapsed

        if wait > 0:
            # Print countdown alerts at key intervals
            alert_at = [600, 300, 60, 30, 10, 5, 3, 2, 1]
            for alert in sorted(alert_at, reverse=True):
                if wait > alert:
                    sleep_for = wait - alert
                    # Sleep in small chunks so Ctrl+C works
                    slept = 0
                    while slept < sleep_for:
                        chunk = min(1.0, sleep_for - slept)
                        time.sleep(chunk)
                        slept += chunk
                    wait = alert
                    remaining = time.monotonic() - t_start
                    next_label = SCHEDULE[idx][0].split("_", 1)[1]
                    print(f"  ⏱  Next shot ({next_label}) in {format_time(alert)}"
                          + " " * 20, end="\r")

            # Final wait
            time.sleep(max(0, wait - 0.1))

        # Beep warning then fire
        beep(1)
        time.sleep(0.1)

        actual_elapsed = time.monotonic() - t_start
        timestamp = datetime.now().strftime("%H%M%S")
        nice_label = label.split("_", 1)[1]   # e.g. "6sec", "1hr"
        filename   = f"{session_id}_{label}_{timestamp}.jpg"
        filepath   = os.path.join(session_dir, filename)

        print(f"  [{idx+1:>2}/{len(SCHEDULE)}]  {nice_label:<8}  →  capturing...", end="\r")

        ok, raw = capture_photo(filepath, session_dir)
        status  = "OK" if ok else "FAILED"

        print(f"  [{idx+1:>2}/{len(SCHEDULE)}]  {nice_label:<8}  {status}   {filename}")
        log.write(
            f"{idx+1:<4} {label:<14} {target_sec:>8.0f} {actual_elapsed:>8.1f}"
            f"  {status:<8}  {filename}\n"
        )
        log.flush()

    # Done
    elapsed_total = time.monotonic() - t_start
    log.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Total elapsed: {format_time(elapsed_total)}\n")
    log.close()

    beep(3)
    print()
    print(f"  Done. {len(SCHEDULE)} photos saved.")
    print(f"  Log: {log_path}")
    print()


def main():
    print()
    print("  Consolidation Test — Dial Gauge Camera")
    print("  Nikon D3300 + digiCamControl")
    print()

    # Check digiCamControl's webserver is reachable
    check_webserver()

    # Pick save folder
    print("  Opening folder picker...")
    folder = pick_folder()
    if not folder:
        print("  No folder selected. Exiting.")
        sys.exit(0)
    print(f"  Save folder: {folder}")

    # Pick load step
    step = pick_load_step()
    if step is None:
        print("  No step selected. Exiting.")
        sys.exit(0)

    info = LOAD_STEPS[step]
    print(f"\n  Step {step}: {info['phase']}  |  {info['kg']} kg")

    # Test camera before committing
    if not test_camera():
        input("\nFix camera connection then press Enter to retry, or Ctrl+C to quit...")
        if not test_camera():
            sys.exit(1)

    # Confirm
    print()
    print(f"  Ready to start Step {step}.")
    print(f"  15 photos will be taken over 24 hours.")
    print()
    print("  Press Enter to begin countdown, or Ctrl+C to cancel.")
    input()

    try:
        run_session(folder, step)
    except KeyboardInterrupt:
        print("\n\n  Session interrupted. Photos taken so far are saved.")

    # Offer to start next step immediately
    print()
    next_step = step + 1
    if next_step in LOAD_STEPS:
        print(f"  Run again for Step {next_step}? Press Y then Enter, or just Enter to quit.")
        ans = input("  > ").strip().lower()
        if ans == "y":
            next_info = LOAD_STEPS[next_step]
            print(f"\n  Next: Step {next_step}  {next_info['phase']}  {next_info['kg']} kg")
            run_session(folder, next_step)

    print("  All done.")
    input("  Press Enter to close.")


if __name__ == "__main__":
    main()