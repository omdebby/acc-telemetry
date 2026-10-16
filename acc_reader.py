"""
acc_reader.py - Record Assetto Corsa Competizione telemetry to CSV.

Run this, then drive in ACC. Press Ctrl+C to stop.
Only samples taken while the session is LIVE are recorded (pause, replay
and menus are skipped).

Each recording produces two files:
    telemetry_YYYYMMDD_HHMMSS.csv        one row per sample
    telemetry_YYYYMMDD_HHMMSS.meta.json  track, car and recording settings

Usage:
    python acc_reader.py
    python acc_reader.py --interval 0.02 --output-dir recordings
    python acc_reader.py --record-all     (also record when not LIVE)
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from pyaccsharedmemory import accSharedMemory


# ---------------------------------------------------------
# Column layout
#
# The CSV header and every row are both built from this one
# list, so they can never fall out of sync.
# ---------------------------------------------------------

COLUMNS = [
    "timestamp",
    "session_time",
    "packet_id",

    # Lap information
    "lap_number",       # laps completed so far (0 = out lap)
    "lap_time",         # current lap time in ms
    "last_lap_time",    # official time of the previous lap in ms
    "is_valid_lap",
    "sector_index",
    "track_position",   # 0.0 - 1.0 around the lap

    # Controls
    "speed_kmh",
    "throttle",
    "brake",
    "steering",
    "gear",             # -1 = reverse, 0 = neutral, 1+ = forward gears
    "rpm",

    # Position on track
    "pos_x",
    "pos_y",
    "pos_z",

    # Vehicle dynamics
    "g_force_x",
    "g_force_y",
    "g_force_z",

    # Fuel
    "fuel",

    # Tyres (FL, FR, RL, RR)
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr",

    # Brakes (FL, FR, RL, RR)
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
]

WHEEL_ORDER = ("front_left", "front_right", "rear_left", "rear_right")
WHEEL_SUFFIXES = ("fl", "fr", "rl", "rr")

# Attribute names differ a little between pyaccsharedmemory versions,
# so each optional field lists the names to try, in order.
TYRE_TEMP_ATTRS = ("tyre_core_temp", "tyre_core_temperature", "tyre_temp")
TYRE_PRESSURE_ATTRS = ("wheel_pressure", "tyre_pressure")
BRAKE_TEMP_ATTRS = ("brake_temp", "brake_temperature")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def first_attr(obj, names, default=None):
    """Return the first attribute from `names` that exists on obj."""
    if obj is None:
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def rounded(value, digits=4):
    """Round numbers to keep the CSV small; pass None through."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def add_wheels(row, prefix, wheels):
    """Write a four-wheel object into prefix_fl / _fr / _rl / _rr columns."""
    for wheel, suffix in zip(WHEEL_ORDER, WHEEL_SUFFIXES):
        row[f"{prefix}_{suffix}"] = rounded(getattr(wheels, wheel, None), 2)


def enum_name(value):
    """Readable name for an enum value (or str() of anything else)."""
    return getattr(value, "name", str(value))


def is_live(graphics):
    """True when the game session is running (not paused/replay/menu)."""
    status = first_attr(graphics, ("status",))
    if status is None:
        return True  # can't tell, so don't block recording
    return "LIVE" in enum_name(status).upper()


def player_coordinates(graphics):
    """Player's world position as (x, y, z), or (None, None, None)."""
    try:
        coords = graphics.car_coordinates[graphics.player_car_id]
        return coords.x, coords.y, coords.z
    except (AttributeError, IndexError, TypeError):
        return None, None, None


def build_row(physics, graphics, current_time, session_time):
    """Turn one shared-memory sample into a dict keyed by COLUMNS."""
    pos_x, pos_y, pos_z = player_coordinates(graphics)

    row = {
        "timestamp": current_time,
        "session_time": rounded(session_time, 3),
        "packet_id": first_attr(physics, ("packet_id",)),

        "lap_number": graphics.completed_lap,
        "lap_time": graphics.current_time,
        "last_lap_time": first_attr(graphics, ("last_time",)),
        "is_valid_lap": first_attr(graphics, ("is_valid_lap",)),
        "sector_index": first_attr(graphics, ("current_sector_index",)),
        "track_position": rounded(
            first_attr(graphics, ("normalized_car_position",)), 5
        ),

        "speed_kmh": rounded(physics.speed_kmh, 2),
        "throttle": rounded(physics.gas),
        "brake": rounded(physics.brake),
        "steering": rounded(physics.steer_angle),

        # ACC reports 0 = reverse, 1 = neutral, 2 = first gear,
        # so subtract 1 to get -1 / 0 / 1+.
        "gear": physics.gear - 1,
        "rpm": physics.rpm,

        "pos_x": rounded(pos_x, 2),
        "pos_y": rounded(pos_y, 2),
        "pos_z": rounded(pos_z, 2),

        "g_force_x": rounded(physics.g_force.x),
        "g_force_y": rounded(physics.g_force.y),
        "g_force_z": rounded(physics.g_force.z),

        "fuel": rounded(physics.fuel, 3),
    }

    add_wheels(row, "tyre_temp", first_attr(physics, TYRE_TEMP_ATTRS))
    add_wheels(row, "tyre_pressure", first_attr(physics, TYRE_PRESSURE_ATTRS))
    add_wheels(row, "brake_temp", first_attr(physics, BRAKE_TEMP_ATTRS))

    return row


def write_metadata(path, static, interval, filename):
    """Save track/car info next to the CSV so recordings are identifiable."""
    meta = {
        "csv_file": filename,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "sample_interval_s": interval,
        "track": first_attr(static, ("track", "track_name")),
        "car_model": first_attr(static, ("car_model",)),
        "columns": COLUMNS,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def report_missing_fields(physics, graphics):
    """Tell the user once which optional fields this library version lacks."""
    checks = {
        "tyre temperatures": first_attr(physics, TYRE_TEMP_ATTRS),
        "tyre pressures": first_attr(physics, TYRE_PRESSURE_ATTRS),
        "brake temperatures": first_attr(physics, BRAKE_TEMP_ATTRS),
        "track position": first_attr(graphics, ("normalized_car_position",)),
        "last lap time": first_attr(graphics, ("last_time",)),
        "lap validity": first_attr(graphics, ("is_valid_lap",)),
    }
    missing = [name for name, value in checks.items() if value is None]

    if player_coordinates(graphics)[0] is None:
        missing.append("car coordinates")

    if missing:
        print("Note: these fields were not found and will be blank:")
        for name in missing:
            print(f"  - {name}")
        print("(Run dir(physics) / dir(graphics) to see the names your")
        print(" pyaccsharedmemory version uses, then adjust the *_ATTRS lists.)")
        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Record ACC telemetry to CSV.")
    parser.add_argument(
        "--interval", type=float, default=0.05,
        help="seconds between samples (default 0.05 = 20 Hz)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="folder for the CSV and metadata files (default: current folder)",
    )
    parser.add_argument(
        "--record-all", action="store_true",
        help="record even when the session is paused, in replay, or in menus",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"telemetry_{stamp}.csv"
    meta_path = output_dir / f"telemetry_{stamp}.meta.json"

    asm = accSharedMemory()

    print("Recording telemetry to:")
    print(f"  {csv_path}")
    print()
    print("Drive in ACC to begin collecting telemetry.")
    print("Press Ctrl+C to stop recording.")
    print()

    start = time.perf_counter()
    next_tick = start
    last_print = 0.0
    last_flush = start
    last_status = None
    sample_count = 0
    checked_fields = False
    waiting_notice_shown = False

    try:
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()

            while True:
                sm = asm.read_shared_memory()

                if sm is None:
                    if not waiting_notice_shown:
                        print("Waiting for ACC shared memory...")
                        waiting_notice_shown = True
                else:
                    physics = sm.Physics
                    graphics = sm.Graphics

                    # Report session state changes (LIVE, PAUSE, REPLAY...)
                    status = enum_name(first_attr(graphics, ("status",), "UNKNOWN"))
                    if status != last_status:
                        print(f"Session status: {status}")
                        last_status = status

                    if args.record_all or is_live(graphics):

                        # First live sample: save metadata, check fields
                        if not checked_fields:
                            write_metadata(
                                meta_path, sm.Static, args.interval, csv_path.name
                            )
                            report_missing_fields(physics, graphics)
                            checked_fields = True

                        now = time.perf_counter()
                        row = build_row(
                            physics, graphics, time.time(), now - start
                        )
                        writer.writerow(row)
                        sample_count += 1

                        # Flush about once a second instead of every sample
                        if now - last_flush >= 1.0:
                            file.flush()
                            last_flush = now

                        # Terminal display twice a second
                        if now - last_print >= 0.5:
                            print(
                                f"Lap: {row['lap_number']:2d} | "
                                f"Speed: {physics.speed_kmh:6.1f} km/h | "
                                f"Throttle: {physics.gas * 100:5.1f}% | "
                                f"Brake: {physics.brake * 100:5.1f}% | "
                                f"Gear: {row['gear']:2d} | "
                                f"RPM: {int(physics.rpm):5d}"
                            )
                            last_print = now

                # Sleep until the next scheduled tick so the sample rate
                # doesn't drift, even though reading takes time.
                next_tick += args.interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()  # fell behind; resync

    except KeyboardInterrupt:
        print("\n")
        print("Recording stopped.")

    finally:
        asm.close()

    print()
    print(f"Saved {sample_count} samples to:")
    print(f"  {csv_path}")
    if checked_fields:
        print(f"  {meta_path}")


if __name__ == "__main__":
    main()