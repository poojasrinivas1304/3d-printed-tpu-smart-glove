import csv
import time
from collections import deque

import serial
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

PORT = "/dev/cu.usbmodem11301"   # Change this to your Arduino Uno port
BAUD = 115200

OUTPUT_CSV = "TPU_backing_5mm_1000.csv"
OUTPUT_FIGURE = "TPU_backing_5mm_1000.png"

# Live plot window length
MAX_SECONDS_ON_PLOT = 120

# Your Arduino sampling interval was 20 ms = 50 Hz
EXPECTED_SAMPLE_RATE_HZ = 50

PLOT_UPDATE_INTERVAL_S = 0.05

# ============================================================


def parse_arduino_line(line):
    """
    Expected Arduino line:
    t_ms,adc,R_ohm,R_kOhm
    1050,412,5050.25,5.0503
    """
    line = line.strip()

    if not line:
        return None

    if line.startswith("t_ms"):
        return None

    parts = line.split(",")

    if len(parts) != 4:
        return None

    try:
        t_ms = float(parts[0])
        adc = float(parts[1])
        r_ohm = float(parts[2])
        r_kohm = float(parts[3])
    except ValueError:
        return None

    return t_ms, adc, r_ohm, r_kohm


def main():
    print("Opening Arduino serial port...")
    print(f"Port: {PORT}")
    print(f"Baud: {BAUD}")

    ser = serial.Serial(PORT, BAUD, timeout=1)

    # Arduino resets when serial opens
    time.sleep(2)
    ser.reset_input_buffer()

    print("Recording UTM cyclic resistance data.")
    print("Press Ctrl+C to stop.")
    print(f"Saving to: {OUTPUT_CSV}")

    max_points = int(MAX_SECONDS_ON_PLOT * EXPECTED_SAMPLE_RATE_HZ * 1.5)

    time_data = deque(maxlen=max_points)
    resistance_data = deque(maxlen=max_points)
    adc_data = deque(maxlen=max_points)

    start_host_time = time.time()
    t0_ms = None
    line_count = 0
    last_plot_update = time.time()

    # Live plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))

    line_obj, = ax.plot([], [], linewidth=1.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resistance (kΩ)")
    ax.set_title("Live UTM Cyclic Resistance Measurement")
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.show(block=False)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "host_elapsed_s",
            "arduino_t_ms",
            "adc",
            "R_ohm",
            "R_kOhm",
        ])

        try:
            while plt.fignum_exists(fig.number):
                raw_line = ser.readline().decode(errors="ignore").strip()
                parsed = parse_arduino_line(raw_line)

                if parsed is None:
                    continue

                t_ms, adc, r_ohm, r_kohm = parsed

                if t0_ms is None:
                    t0_ms = t_ms

                arduino_elapsed_s = (t_ms - t0_ms) / 1000.0
                host_elapsed_s = time.time() - start_host_time

                writer.writerow([
                    host_elapsed_s,
                    t_ms,
                    adc,
                    r_ohm,
                    r_kohm,
                ])

                line_count += 1

                if line_count % EXPECTED_SAMPLE_RATE_HZ == 0:
                    f.flush()
                    print(
                        f"t={arduino_elapsed_s:8.2f}s | "
                        f"ADC={adc:5.0f} | "
                        f"R={r_kohm:8.4f} kΩ"
                    )

                time_data.append(arduino_elapsed_s)
                resistance_data.append(r_kohm)
                adc_data.append(adc)

                # Update plot
                if time.time() - last_plot_update >= PLOT_UPDATE_INTERVAL_S:
                    t_list = list(time_data)
                    r_list = list(resistance_data)

                    if len(t_list) > 1:
                        t_max = t_list[-1]
                        t_min = max(0, t_max - MAX_SECONDS_ON_PLOT)

                        line_obj.set_data(t_list, r_list)

                        ax.set_xlim(t_min, max(t_min + 1, t_max))
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)

                        fig.canvas.draw_idle()
                        plt.pause(0.001)

                    last_plot_update = time.time()

        except KeyboardInterrupt:
            print()
            print("Stopped recording.")

        finally:
            ser.close()
            f.flush()
            fig.savefig(OUTPUT_FIGURE, dpi=200)

            print()
            print(f"Saved CSV: {OUTPUT_CSV}")
            print(f"Saved last plot view: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()