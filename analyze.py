"""
analyze.py - Compare two laps from an acc_reader.py telemetry recording.

By default it compares the most recent complete lap against your fastest
valid lap. Lap numbers are the `lap_number` column from the CSV
(laps completed so far, so 0 is the out lap).

Usage:
    python analyze.py                          (newest telemetry_*.csv)
    python analyze.py telemetry_2026....csv
    python analyze.py --lap 5 --reference 3
    python analyze.py --corners 16             (tune corner detection to
                                                a track with 16 corners)
    python analyze.py --save-dir plots --no-show

Works with new recordings (uses track_position, is_valid_lap and
last_lap_time) and falls back to estimating distance from speed for old
recordings that don't have those columns.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# SETTINGS
# ==========================================

REQUIRED_COLUMNS = [
    "timestamp", "lap_number", "lap_time",
    "speed_kmh", "throttle", "brake", "steering",
]

# Lap detection
RESET_JUMP_MS = 1000       # a drop bigger than this in lap_time = timer reset
START_TOLERANCE_MS = 2000  # a full lap must start within this of the line
START_POSITION_MAX = 0.03  # ...and (if known) within 3% of the start line
END_POSITION_MIN = 0.97    # a full lap must end within 3% of the line
MIN_SAMPLES = 50

# Resampling
GRID_POINTS = 1000         # points used to compare laps at equal progress

# Braking detection
BRAKE_THRESHOLD = 0.10     # brake input that counts as braking
MIN_BRAKE_S = 0.20         # ignore brake taps shorter than this
MERGE_GAP_S = 0.50         # join brake events separated by less than this

# Corner detection (from steering input, so corners taken flat or with
# only a lift are found too, not just corners with a braking zone)
STEER_FRACTION = 0.12      # turning = |steering| above this share of the
                           # lap's typical maximum (lower = more corners)
STEER_SMOOTH_S = 0.30      # smooth steering over this long before detecting
MIN_CORNER_S = 0.40        # ignore steering corrections shorter than this
CORNER_MERGE_S = 0.50      # join same-direction turns closer together than this
BRAKE_LEAD_M = 150         # a braking zone this far before a corner belongs to it
ZONE_LEAD_M = 30           # corner zone starts this far before a corner with no braking
TUNE_FRACTIONS = [0.30, 0.25, 0.20, 0.16, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]

# Track map
MIRROR_MAP = True          # flip x so left/right turns match the real track
                           # (use --no-mirror-map to turn off)

# Corner analysis
EXIT_50_M = 50
EXIT_100_M = 100
THROTTLE_PICKUP = 0.20
STRONG_THROTTLE = 0.80

# Coaching thresholds
BRAKE_POINT_MIN_M = 5
SPEED_MIN_KMH = 2
PICKUP_MIN_M = 5
TIME_CHANGE_MIN_S = 0.03

METRICS = [
    "brake_start", "min_speed", "throttle_pickup",
    "strong_throttle", "speed_50", "speed_100", "exit_throttle",
]


# ==========================================
# SMALL HELPERS
# ==========================================

def fmt_ms(ms):
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}:{seconds:06.3f}"


def fmt_diff(value, unit, digits=1):
    if pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}f} {unit}"


def to_bool(value):
    """Parse a CSV validity flag; None when unknown."""
    if pd.isna(value):
        return None
    return {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}.get(str(value).strip().lower())


def runs_of_true(mask):
    """(start, end) index pairs, inclusive, for each run of True."""
    padded = np.concatenate(([False], mask, [False])).astype(int)
    change = np.diff(padded)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return list(zip(starts, ends))


def has_track_position(data):
    return (
        "track_position" in data.columns
        and data["track_position"].notna().any()
    )


# ==========================================
# LOADING AND LAP DETECTION
# ==========================================

@dataclass
class LapInfo:
    number: int
    data: pd.DataFrame
    time_ms: float      # official time if known, otherwise measured
    valid: object       # True / False / None (unknown)
    complete: bool


def load_telemetry(path):
    data = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    for column in set(REQUIRED_COLUMNS + ["track_position", "last_lap_time", "pos_x", "pos_z"]):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED_COLUMNS).copy()
    if data.empty:
        raise ValueError("No usable telemetry samples found.")
    if (data["lap_number"].diff() < 0).any():
        raise ValueError("Recording contains a session restart. Split into sessions before comparing.")
    if (data["lap_number"] % 1 != 0).any() or (data["lap_time"] < 0).any():
        raise ValueError("Lap numbers must be integers and times non-negative.")
    if "track_position" in data:
        data.loc[~data["track_position"].between(0, 1), "track_position"] = np.nan
    data["lap_number"] = data["lap_number"].astype(int)
    return data


def official_time_ms(data, lap_number):
    """Official time of a lap, read from the next lap's last_lap_time."""
    if "last_lap_time" not in data.columns:
        return None

    following = data[data["lap_number"] == lap_number + 1]
    following = following.sort_values("timestamp")
    values = following["last_lap_time"].dropna()
    values = values[(values > 0) & (values < 2147483647)]

    if values.empty:
        return None

    # Skip the first few samples in case the value hasn't updated yet
    return float(values.iloc[: 40].iloc[-1])


def find_laps(data):
    """Split the recording into laps and decide which are complete."""
    use_position = has_track_position(data)
    laps = []

    for number, group in data.groupby("lap_number", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)

        # The lap timer can reset slightly before or after lap_number
        # changes, so split on resets and keep the longest run.
        lap_time = group["lap_time"].to_numpy()
        resets = np.where(np.diff(lap_time) < -RESET_JUMP_MS)[0] + 1
        segments = np.split(np.arange(len(group)), resets)
        best = max(segments, key=len)

        segment = group.iloc[best].drop_duplicates("lap_time")
        segment = segment.reset_index(drop=True)

        if len(segment) < MIN_SAMPLES:
            continue

        start_ok = segment["lap_time"].iloc[0] <= START_TOLERANCE_MS
        end_ok = None  # decided later when track position is unknown

        if use_position:
            start_pos = segment["track_position"].iloc[0]
            end_pos = segment["track_position"].iloc[-1]
            if pd.notna(start_pos):
                start_ok = start_ok and start_pos <= START_POSITION_MAX
            if pd.notna(end_pos):
                end_ok = end_pos >= END_POSITION_MIN

        measured_ms = float(segment["lap_time"].iloc[-1])
        official = official_time_ms(data, number)

        valid = None
        if "is_valid_lap" in segment.columns:
            valid = to_bool(segment["is_valid_lap"].iloc[-1])

        laps.append({
            "number": number,
            "segment": segment,
            "time_ms": official if official else measured_ms,
            "valid": valid,
            "start_ok": start_ok,
            "end_ok": end_ok,
            "measured_ms": measured_ms,
            "crossed_line": bool((data["lap_number"] == number + 1).any()),
        })

    # Without track position, call a lap complete if it started at the
    # line and lasted nearly as long as a typical lap.
    starters = [lap["measured_ms"] for lap in laps if lap["start_ok"]]
    typical = np.median(starters) if starters else None

    results = []
    for lap in laps:
        if lap["end_ok"] is None:
            end_ok = typical is not None and lap["measured_ms"] >= 0.9 * typical
        else:
            end_ok = lap["end_ok"]

        results.append(LapInfo(
            number=lap["number"],
            data=lap["segment"],
            time_ms=lap["time_ms"],
            valid=lap["valid"],
            complete=bool(lap["start_ok"] and end_ok and lap["crossed_line"]),
        ))

    return results


def choose_laps(laps, lap_arg, reference_arg):
    complete = {lap.number: lap for lap in laps if lap.complete}

    if not complete:
        raise ValueError("No complete laps found in this recording.")

    # Reference: fastest valid lap (fastest of any if none are valid)
    if reference_arg is not None:
        if reference_arg not in complete:
            raise ValueError(f"Lap {reference_arg} is not a complete lap.")
        reference = complete[reference_arg]
    else:
        valid = [lap for lap in complete.values() if lap.valid is True]
        if not valid:
            valid = [lap for lap in complete.values() if lap.valid is None]
        pool = valid if valid else list(complete.values())
        reference = min(pool, key=lambda lap: lap.time_ms)

    # Lap to analyze: most recent complete lap that isn't the reference
    if lap_arg is not None:
        if lap_arg not in complete:
            raise ValueError(f"Lap {lap_arg} is not a complete lap.")
        chosen = complete[lap_arg]
    else:
        others = [n for n in complete if n != reference.number]
        if not others:
            raise ValueError("Need at least two complete laps to compare.")
        chosen = complete[max(others)]

    return chosen, reference


# ==========================================
# BUILDING LAPS ON A COMMON DISTANCE AXIS
# ==========================================

def build_lap(info, use_position, length_m=None):
    """
    Add time, progress (0-1 around the lap) and distance columns.

    Progress comes from the game's track position when available,
    otherwise from integrating speed and normalizing to the lap's own
    total, which removes most integration drift.
    """
    lap = info.data.copy()

    lap["t"] = lap["lap_time"] / 1000
    dt = lap["t"].diff().fillna(0)
    speed = lap["speed_kmh"].clip(lower=0) / 3.6
    lap["raw_dist"] = ((speed + speed.shift(fill_value=speed.iloc[0])) / 2 * dt).cumsum()
    positioned = use_position and "track_position" in lap and lap["track_position"].notna().all()
    progress = lap["track_position"].to_numpy(float) if positioned else lap["raw_dist"].to_numpy(float)
    previous_max = np.concatenate(([-np.inf], np.maximum.accumulate(progress)[:-1]))
    keep = progress > previous_max
    lap = lap.loc[keep].reset_index(drop=True)
    progress = progress[keep]
    if len(progress) < 2 or progress[-1] <= progress[0]:
        raise ValueError("Lap has no usable forward progress.")
    raw_length = float(lap["raw_dist"].iloc[-1])
    if positioned:
        raw_length /= progress[-1] - progress[0]
    else:
        progress = (progress - progress[0]) / (progress[-1] - progress[0])
    lap["progress"] = progress
    lap.attrs["raw_length_m"] = raw_length
    lap.attrs["official_time_s"] = info.time_ms / 1000
    lap.attrs["positioned"] = positioned
    if length_m is not None:
        lap["dist_m"] = progress * length_m

    return lap


def on_grid(lap, column, grid):
    if column == "t" and lap.attrs.get("positioned"):
        inside = lap["progress"].between(0, 1, inclusive="neither")
        return np.interp(grid, np.r_[0, lap.loc[inside, "progress"], 1],
                         np.r_[0, lap.loc[inside, "t"], lap.attrs["official_time_s"]])
    return np.interp(grid, lap["progress"], lap[column])


# ==========================================
# LAP TABLE AND TIME DELTA
# ==========================================

def print_lap_table(laps):
    print("Laps found:")
    for lap in laps:
        if not lap.complete:
            continue
        flag = {True: "", False: "  (invalid)", None: ""}[lap.valid]
        print(f"  Lap {lap.number}: {fmt_ms(lap.time_ms)}{flag}")

    skipped = [str(lap.number) for lap in laps if not lap.complete]
    if skipped:
        print(f"  Skipped incomplete laps: {', '.join(skipped)}")
    print()


def compute_delta(lap, ref, grid):
    """Time difference at equal progress. Positive = lap is behind."""
    return on_grid(lap, "t", grid) - on_grid(ref, "t", grid)


# ==========================================
# SECTION ANALYSIS
# ==========================================

def analyze_sections(lap, ref, num_sections):
    if num_sections < 1:
        raise ValueError("Sections must be positive.")
    edges = np.linspace(0, 1, num_sections + 1)
    rows = []

    for i in range(num_sections):
        lo, hi = edges[i], edges[i + 1]
        times = {}
        speeds = {}

        for name, data in (("lap", lap), ("ref", ref)):
            t_lo = on_grid(data, "t", lo)
            t_hi = on_grid(data, "t", hi)
            times[name] = t_hi - t_lo

            in_section = data[
                (data["progress"] >= lo) & (data["progress"] <= hi)
            ]
            speeds[name] = in_section["speed_kmh"].mean()

        rows.append({
            "section": i + 1,
            "ref_time": times["ref"],
            "lap_time": times["lap"],
            "time_difference": times["lap"] - times["ref"],
            "speed_difference": speeds["lap"] - speeds["ref"],
        })

    return pd.DataFrame(rows)


# ==========================================
# BRAKING EVENTS
# ==========================================

def detect_braking(lap):
    columns = [
        "start_dist", "end_dist", "duration",
        "max_brake", "start_speed", "end_speed",
    ]

    brake = lap["brake"].to_numpy()
    t = lap["t"].to_numpy()
    speed = lap["speed_kmh"].to_numpy()
    dist = lap["dist_m"].to_numpy()

    merged = []
    for start, end in runs_of_true(brake >= BRAKE_THRESHOLD):
        if merged and t[start] - t[merged[-1][1]] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    events = []
    for start, end in merged:
        duration = t[end] - t[start]
        if duration < MIN_BRAKE_S:
            continue
        events.append({
            "start_dist": dist[start],
            "end_dist": dist[end],
            "duration": duration,
            "max_brake": brake[start:end + 1].max(),
            "start_speed": speed[start],
            "end_speed": speed[end],
        })

    return pd.DataFrame(events, columns=columns)


def find_brake_event(events, lo, hi, claimed):
    """
    First unclaimed braking event overlapping the stretch lo..hi (metres).
    Each event can belong to only one corner, so a short chicane's single
    braking zone goes to its first corner.
    """
    for k, event in events.iterrows():
        if k in claimed:
            continue
        if event["end_dist"] >= lo and event["start_dist"] <= hi:
            claimed.add(k)
            return k
    return None


# ==========================================
# CORNER DETECTION (from steering)
# ==========================================

CORNER_COLUMNS = [
    "corner", "start_dist", "end_dist", "apex_dist", "apex_speed",
    "zone_start", "brake_start", "brake_end",
]


def detect_corners(ref, ref_events, steer_fraction=STEER_FRACTION):
    """
    Find corners on the reference lap from steering input.

    A corner is a stretch where steering stays on one side beyond a
    threshold. Left-right sequences (chicanes, S-bends) split into
    separate corners because the steering changes sign. Braking zones
    are attached afterwards as each corner's entry, so corners with
    no braking (flat-out or lift-only) are still counted.
    """
    if not 0 < steer_fraction <= 1:
        raise ValueError("Steering threshold must be between 0 and 1.")
    t = ref["t"].to_numpy()
    dist = ref["dist_m"].to_numpy()
    speed = ref["speed_kmh"].to_numpy()

    step = np.median(np.diff(t)) if len(t) > 1 else 0.05
    window = max(1, int(round(STEER_SMOOTH_S / step)))
    steer = (
        ref["steering"].rolling(window, center=True, min_periods=1)
        .mean().to_numpy()
    )

    scale = np.percentile(np.abs(steer), 98)
    if scale <= 0:
        return pd.DataFrame(columns=CORNER_COLUMNS)

    limit = steer_fraction * scale
    side = np.where(steer > limit, 1, np.where(steer < -limit, -1, 0))

    # Runs of steering on one side, joining same-side runs that are
    # only briefly interrupted
    runs = []
    i = 0
    while i < len(side):
        if side[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(side) and side[j + 1] == side[i]:
            j += 1
        if (runs and runs[-1][2] == side[i]
                and t[i] - t[runs[-1][1]] < CORNER_MERGE_S):
            runs[-1][1] = j
        else:
            runs.append([i, j, int(side[i])])
        i = j + 1

    rows = []
    claimed = set()
    for start, end, _ in runs:
        if t[end] - t[start] < MIN_CORNER_S:
            continue

        apex = start + int(np.argmax(np.abs(steer[start:end + 1])))
        start_dist = dist[start]
        end_dist = max(dist[end], dist[apex])

        k = find_brake_event(
            ref_events, start_dist - BRAKE_LEAD_M, dist[apex], claimed
        )
        if k is None:
            brake_start = brake_end = np.nan
            zone_start = max(start_dist - ZONE_LEAD_M, dist[0])
        else:
            brake_start = ref_events.loc[k, "start_dist"]
            brake_end = ref_events.loc[k, "end_dist"]
            zone_start = min(brake_start, start_dist)

        rows.append({
            "corner": len(rows) + 1,
            "start_dist": start_dist,
            "end_dist": end_dist,
            "apex_dist": dist[apex],
            "apex_speed": speed[apex],
            "zone_start": zone_start,
            "brake_start": brake_start,
            "brake_end": brake_end,
        })

    return pd.DataFrame(rows, columns=CORNER_COLUMNS)


def tune_corners(ref, ref_events, expected):
    """Try a range of thresholds; keep the one closest to `expected`."""
    if expected < 1:
        raise ValueError("Expected corners must be positive.")
    best = None
    for fraction in TUNE_FRACTIONS:
        found = detect_corners(ref, ref_events, fraction)
        miss = abs(len(found) - expected)
        # '<' keeps the larger (stricter) threshold on a tie
        if best is None or miss < best[0]:
            best = (miss, fraction, found)
    return best[2], best[1]



def load_profile(csv_path, profile_path=None):
    if profile_path:
        return validate_profile(json.loads(Path(profile_path).read_text(encoding="utf-8")))
    meta = Path(csv_path).with_suffix(".meta.json")
    if meta.exists():
        track = str(json.loads(meta.read_text(encoding="utf-8")).get("track", "")).split("\0")[0].strip().lower()
        if track == "barcelona":
            return load_profile(csv_path, Path(__file__).parent / "profiles" / "barcelona.json")
    return None


def validate_profile(profile):
    turns = profile.get("turns", [])
    if not turns:
        raise ValueError("A corner profile needs at least one turn.")
    previous = 0
    for i, turn in enumerate(turns, 1):
        start, apex, end = (float(turn[k]) for k in ("start", "apex", "end"))
        if not (0 <= start < apex < end <= 1) or start < previous:
            raise ValueError(f"T{i}: require 0 <= start < apex < end <= 1, in order without overlap.")
        previous = end
    return profile


def profile_corners(ref, events, profile):
    validate_profile(profile)
    if not ref.attrs.get("positioned"):
        raise ValueError("Profiles need complete track_position data. Use automatic detection for legacy CSVs.")
    length = ref.attrs["raw_length_m"]
    rows, claimed = [], set()
    for i, turn in enumerate(profile["turns"], 1):
        start, apex, end = [float(turn[k]) * length for k in ("start", "apex", "end")]
        k = find_brake_event(events, start - BRAKE_LEAD_M, apex, claimed)
        brake_start = events.loc[k, "start_dist"] if k is not None else np.nan
        rows.append(dict(corner=i, start_dist=start, apex_dist=apex, end_dist=end,
                         apex_speed=np.interp(apex, ref["dist_m"], ref["speed_kmh"]),
                         zone_start=min(start, brake_start) if k is not None else max(0, start-ZONE_LEAD_M),
                         brake_start=brake_start,
                         brake_end=events.loc[k, "end_dist"] if k is not None else np.nan))
    return pd.DataFrame(rows, columns=CORNER_COLUMNS)


def print_corner_list(corners, fraction):
    print("==========================================")
    print(f"CORNERS: {len(corners)}")
    print("==========================================")
    for _, c in corners.iterrows():
        braking = (
            f"braking from {c['brake_start']:.0f} m"
            if pd.notna(c["brake_start"]) else "no braking"
        )
        print(
            f"Corner {int(c['corner']):2d} | "
            f"{c['start_dist']:5.0f}-{c['end_dist']:5.0f} m | "
            f"apex {c['apex_speed']:5.1f} km/h | {braking}"
        )
    print()


# ==========================================
# CORNER ANALYSIS
# ==========================================

def window_metrics(lap, zone_start, window_end, span_start, span_end):
    """
    Measure a lap over the same stretch of track as the reference.

    zone_start..window_end is the whole corner zone; span_start..span_end
    is where the steering was turned, which is where the apex is searched
    for. Exit measurements are taken relative to span_end.
    """
    out = dict.fromkeys(METRICS, np.nan)

    window = lap[
        (lap["dist_m"] >= zone_start) & (lap["dist_m"] <= window_end)
    ]
    if len(window) < 3:
        return out

    span = window[
        (window["dist_m"] >= span_start) & (window["dist_m"] <= span_end)
    ]
    if len(span) < 3:
        span = window

    apex = span["speed_kmh"].idxmin()
    out["min_speed"] = window.loc[apex, "speed_kmh"]
    apex_dist = window.loc[apex, "dist_m"]
    ref_end = span_end

    after_apex = window[window["dist_m"] >= apex_dist]

    pickup = after_apex[after_apex["throttle"] >= THROTTLE_PICKUP]
    if len(pickup):
        out["throttle_pickup"] = pickup["dist_m"].iloc[0]

    strong = after_apex[after_apex["throttle"] >= STRONG_THROTTLE]
    if len(strong):
        out["strong_throttle"] = strong["dist_m"].iloc[0]

    out["speed_50"] = np.interp(
        ref_end + EXIT_50_M, lap["dist_m"], lap["speed_kmh"], left=np.nan, right=np.nan
    )
    out["speed_100"] = np.interp(
        ref_end + EXIT_100_M, lap["dist_m"], lap["speed_kmh"], left=np.nan, right=np.nan
    )

    exit_zone = lap[
        (lap["dist_m"] >= ref_end)
        & (lap["dist_m"] <= ref_end + EXIT_100_M)
    ]
    if len(exit_zone):
        out["exit_throttle"] = exit_zone["throttle"].mean() * 100

    return out


def analyze_corners(lap, ref, lap_events, corners, grid_dist, delta):
    """One row per reference corner, comparing lap against ref."""
    length = ref["dist_m"].iloc[-1]
    claimed = set()
    rows = []

    for _, c in corners.iterrows():
        window_start = c["zone_start"]
        window_end = min(c["end_dist"] + EXIT_100_M, length)

        ref_vals = window_metrics(
            ref, window_start, window_end, c["start_dist"], c["end_dist"]
        )
        lap_vals = window_metrics(
            lap, window_start, window_end, c["start_dist"], c["end_dist"]
        )

        # Braking points only compare when the reference braked here
        ref_vals["brake_start"] = c["brake_start"]
        lap_vals["brake_start"] = np.nan
        if pd.notna(c["brake_start"]):
            k = find_brake_event(
                lap_events, c["start_dist"] - BRAKE_LEAD_M,
                c["apex_dist"], claimed,
            )
            if k is not None:
                lap_vals["brake_start"] = lap_events.loc[k, "start_dist"]

        row = {
            "corner": int(c["corner"]),
            "distance": window_start,
            "time_change": (
                np.interp(window_end, grid_dist, delta)
                - np.interp(window_start, grid_dist, delta)
            ),
        }
        for key in METRICS:
            row[f"{key}_ref"] = ref_vals[key]
            row[f"{key}_lap"] = lap_vals[key]
            row[f"{key}_diff"] = lap_vals[key] - ref_vals[key]

        rows.append(row)

    return pd.DataFrame(rows)


def print_corner_comparison(corners, lap_no, ref_no):
    print("==========================================")
    print(f"CORNER COMPARISON - Lap {lap_no} vs Lap {ref_no}")
    print("==========================================")

    for _, c in corners.iterrows():
        print()
        print(f"Corner {int(c['corner'])} (entry at ~{c['distance']:.0f} m)")
        print(f"  Time in corner:   {fmt_diff(c['time_change'], 's', 3)}")
        print(f"  Braking point:    {fmt_diff(c['brake_start_diff'], 'm')}")
        print(f"  Minimum speed:    {fmt_diff(c['min_speed_diff'], 'km/h')}")
        print(f"  Throttle pickup:  {fmt_diff(c['throttle_pickup_diff'], 'm')}")
        print(f"  80% throttle:     {fmt_diff(c['strong_throttle_diff'], 'm')}")
        print(f"  Speed +50m:       {fmt_diff(c['speed_50_diff'], 'km/h')}")
        print(f"  Speed +100m:      {fmt_diff(c['speed_100_diff'], 'km/h')}")
        print(f"  Exit throttle:    {fmt_diff(c['exit_throttle_diff'], '%')}")


def print_coaching(corners, lap_no, ref_no):
    """Plain-language summary, biggest time loss first."""
    print()
    print("==========================================")
    print(f"WHERE LAP {lap_no} GAINED OR LOST TIME vs LAP {ref_no}")
    print("==========================================")

    printed = False
    for _, c in corners.sort_values("time_change", ascending=False).iterrows():
        notes = []

        diff = c["brake_start_diff"]
        if pd.notna(diff) and abs(diff) >= BRAKE_POINT_MIN_M:
            word = "earlier" if diff < 0 else "later"
            notes.append(f"braked {abs(diff):.0f} m {word}")

        diff = c["min_speed_diff"]
        if pd.notna(diff) and abs(diff) >= SPEED_MIN_KMH:
            word = "lower" if diff < 0 else "higher"
            notes.append(f"minimum speed {abs(diff):.1f} km/h {word}")

        diff = c["throttle_pickup_diff"]
        if pd.notna(diff) and abs(diff) >= PICKUP_MIN_M:
            word = "later" if diff > 0 else "earlier"
            notes.append(f"picked up throttle {abs(diff):.0f} m {word}")

        diff = c["speed_100_diff"]
        if pd.notna(diff) and abs(diff) >= SPEED_MIN_KMH:
            word = "slower" if diff < 0 else "faster"
            notes.append(f"{abs(diff):.1f} km/h {word} 100 m after the zone")

        change = c["time_change"]
        if not notes and abs(change) < TIME_CHANGE_MIN_S:
            continue

        verdict = "lost" if change > 0 else "gained"
        detail = "; ".join(notes) if notes else "no big input differences"
        print(
            f"Corner {int(c['corner'])} (~{c['distance']:.0f} m): "
            f"{verdict} {abs(change):.3f} s - {detail}"
        )
        printed = True

    if not printed:
        print("No significant differences found.")


# ==========================================
# PLOTS
# ==========================================

def plot_overview(lap, ref, lap_no, ref_no, grid_dist, delta, corners):
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5, 1.5]},
    )
    speed_ax, delta_ax, throttle_ax, brake_ax, steer_ax = axes

    for data, label, color in (
        (ref, f"Lap {ref_no} (reference)", "tab:blue"),
        (lap, f"Lap {lap_no}", "tab:orange"),
    ):
        speed_ax.plot(data["dist_m"], data["speed_kmh"], label=label, color=color)
        throttle_ax.plot(data["dist_m"], data["throttle"] * 100, color=color)
        brake_ax.plot(data["dist_m"], data["brake"] * 100, color=color)
        steer_ax.plot(data["dist_m"], data["steering"], color=color)

    delta_ax.plot(grid_dist, delta, color="tab:red")
    delta_ax.axhline(0, linestyle="--", color="black", linewidth=0.8)

    speed_ax.set_ylabel("Speed (km/h)")
    delta_ax.set_ylabel("Time delta (s)\n+ = behind")
    throttle_ax.set_ylabel("Throttle (%)")
    brake_ax.set_ylabel("Brake (%)")
    steer_ax.set_ylabel("Steering")
    steer_ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
    steer_ax.set_xlabel("Distance (m, estimated)")

    # Shade each detected corner and number it at its apex
    for _, c in corners.iterrows():
        for ax in axes:
            ax.axvspan(
                c["start_dist"], c["end_dist"], color="grey", alpha=0.12
            )
        speed_ax.annotate(
            f"T{int(c['corner'])}",
            xy=(c["apex_dist"], 1.01 + (int(c["corner"]) % 2) * 0.07),
            xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=8,
        )

    for ax in axes:
        ax.grid(True, alpha=0.3)
    speed_ax.legend(loc="lower right")
    speed_ax.set_title(f"Lap {lap_no} vs Lap {ref_no}", pad=34)
    fig.tight_layout()
    return fig


def plot_sections(sections, lap_no, ref_no):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "tab:red" if v > 0 else "tab:green"
        for v in sections["time_difference"]
    ]
    ax.bar(sections["section"], sections["time_difference"], color=colors)
    ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
    ax.set_title(
        f"Section time difference - Lap {lap_no} vs Lap {ref_no} "
        "(red = slower, green = faster)"
    )
    ax.set_xlabel("Section")
    ax.set_ylabel("Time difference (s)")
    ax.set_xticks(sections["section"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_corners(corners, lap_no, ref_no):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    panels = [
        (axes[0, 0], "time_change", "Time in corner (s)", True),
        (axes[0, 1], "min_speed_diff", "Minimum speed (km/h)", False),
        (axes[1, 0], "speed_100_diff", "Speed +100 m after zone (km/h)", False),
        (axes[1, 1], "throttle_pickup_diff", "Throttle pickup (m, + = later)", True),
    ]

    for ax, column, title, positive_is_bad in panels:
        values = corners[column]
        if positive_is_bad:
            colors = ["tab:red" if v > 0 else "tab:green" for v in values]
        else:
            colors = ["tab:green" if v > 0 else "tab:red" for v in values]
        ax.bar(corners["corner"], values, color=colors)
        ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Corner")
    axes[1, 0].set_xticks(corners["corner"])
    fig.suptitle(f"Corner differences - Lap {lap_no} vs Lap {ref_no}")
    fig.tight_layout()
    return fig


def plot_track_map(ref, corners, ref_no, mirror=MIRROR_MAP):
    fig, ax = plt.subplots(figsize=(9, 9))

    # The game's world axes come out mirrored on a normal plot, so flip x
    # (see MIRROR_MAP). Every x used below goes through the same flip.
    px = -ref["pos_x"] if mirror else ref["pos_x"]
    pz = ref["pos_z"]

    points = ax.scatter(px, pz, c=ref["speed_kmh"], s=6, cmap="viridis")
    fig.colorbar(points, ax=ax, label="Speed (km/h)")

    # Start/finish marker and an arrow showing the direction of travel,
    # so it's easy to check the map against the real track
    ahead = min(
        int(np.searchsorted(ref["dist_m"].to_numpy(), 150)), len(ref) - 1
    )
    ax.annotate(
        "", xy=(px.iloc[ahead], pz.iloc[ahead]), xytext=(px.iloc[0], pz.iloc[0]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
    )
    ax.scatter([px.iloc[0]], [pz.iloc[0]], marker="*", s=250,
               color="gold", edgecolor="black", zorder=5)
    ax.annotate("S/F", xy=(px.iloc[0], pz.iloc[0]), xytext=(8, 8),
                textcoords="offset points", fontsize=10, fontweight="bold")

    for _, c in corners.iterrows():
        x = np.interp(c["apex_dist"], ref["dist_m"], px)
        z = np.interp(c["apex_dist"], ref["dist_m"], pz)
        ax.annotate(
            f"T{int(c['corner'])}", xy=(x, z), xytext=(14 if int(c["corner"]) % 2 else -22, 18 if int(c["corner"]) % 2 else -20),
            textcoords="offset points", arrowprops=dict(arrowstyle="-", color="black"),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black"),
        )

    ax.set_aspect("equal")
    ax.set_title(f"Track map (Lap {ref_no}) - turn labels")
    ax.set_xlabel("X (mirrored)" if mirror else "X")
    ax.set_ylabel("Z")
    fig.tight_layout()
    return fig


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two ACC laps.")
    parser.add_argument(
        "csv", nargs="?",
        help="telemetry CSV (default: newest telemetry_*.csv here)",
    )
    parser.add_argument("--profile", help="corner profile JSON")
    parser.add_argument("--auto-corners", action="store_true", help="use estimated steering corners")
    parser.add_argument("--lap", type=int, help="lap to analyze")
    parser.add_argument("--reference", type=int, help="lap to compare against")
    parser.add_argument("--sections", type=int, default=10)
    parser.add_argument(
        "--corners", type=int,
        help="how many corners the track has (e.g. 16 for the Nurburgring "
             "GP circuit); the steering threshold is tuned to match",
    )
    parser.add_argument(
        "--steer-fraction", type=float, default=STEER_FRACTION,
        help=f"steering threshold for corner detection "
             f"(default {STEER_FRACTION}; lower finds more corners)",
    )
    parser.add_argument("--save-dir", help="save plots as PNGs in this folder")
    parser.add_argument("--no-mirror-map", action="store_true",
                        help="draw the track map without flipping x")
    parser.add_argument("--no-show", action="store_true",
                        help="don't open plot windows")
    return parser.parse_args()


def find_default_csv():
    files = sorted(Path(".").glob("telemetry_*.csv"))
    if not files:
        raise ValueError("No telemetry_*.csv found. Pass a CSV path.")
    return files[-1]


def main():
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else find_default_csv()
    print(f"Loading {csv_path}")
    print()

    data = load_telemetry(csv_path)
    use_position = has_track_position(data)
    if not use_position:
        print("Note: no track_position column (older recording), so distance")
        print("is estimated from speed. Re-record with the new acc_reader.py")
        print("for more accurate comparisons.")
        print()

    laps = find_laps(data)
    print_lap_table(laps)

    lap_info, ref_info = choose_laps(laps, args.lap, args.reference)
    lap_no, ref_no = lap_info.number, ref_info.number
    print(
        f"Comparing Lap {lap_no} ({fmt_ms(lap_info.time_ms)}) "
        f"against Lap {ref_no} ({fmt_ms(ref_info.time_ms)})"
    )
    print()

    # Build both laps on the same distance axis. The reference lap's
    # length sets the scale, so both laps share identical distances.
    use_position = all("track_position" in info.data and info.data["track_position"].notna().all()
                       for info in (lap_info, ref_info))
    ref = build_lap(ref_info, use_position)
    length_m = ref.attrs["raw_length_m"]
    ref["dist_m"] = ref["progress"] * length_m
    lap = build_lap(lap_info, use_position, length_m)

    grid = np.linspace(0, 1, GRID_POINTS)
    grid_dist = grid * length_m
    delta = compute_delta(lap, ref, grid)

    print(f"Estimated lap length: {length_m:.0f} m")
    print(f"Final delta (official endpoints): {delta[-1]:+.3f} s")
    print(
        f"Official time difference:        "
        f"{(lap_info.time_ms - ref_info.time_ms) / 1000:+.3f} s"
    )
    print()

    # Sections
    sections = analyze_sections(lap, ref, args.sections)
    print("==========================================")
    print("SECTION ANALYSIS")
    print("==========================================")
    for _, row in sections.iterrows():
        print(
            f"Section {int(row['section']):2d} | "
            f"Lap {ref_no}: {row['ref_time']:6.3f}s | "
            f"Lap {lap_no}: {row['lap_time']:6.3f}s | "
            f"Difference: {row['time_difference']:+6.3f}s"
        )
    print()

    # Braking and corners
    ref_events = detect_braking(ref)
    lap_events = detect_braking(lap)
    print(
        f"Braking zones: {len(ref_events)} on Lap {ref_no}, "
        f"{len(lap_events)} on Lap {lap_no}"
    )
    print()

    # Corners come from steering, with braking zones attached as entries
    profile = None if args.auto_corners else load_profile(csv_path, args.profile)
    if profile is not None and ref.attrs.get("positioned"):
        ref_corners = profile_corners(ref, ref_events, profile)
        fraction = args.steer_fraction
        print(f"Corner profile: {profile['name']}")
    elif args.corners:
        ref_corners, fraction = tune_corners(ref, ref_events, args.corners)
    else:
        fraction = args.steer_fraction
        ref_corners = detect_corners(ref, ref_events, fraction)
    print_corner_list(ref_corners, fraction)

    if args.corners and len(ref_corners) != args.corners:
        print(
            f"Note: closest match is {len(ref_corners)} corners, not "
            f"{args.corners}. Check the numbered shading on the overview "
            "plot to see which corners are missing or split."
        )
        print()

    figures = {}
    figures["overview"] = plot_overview(
        lap, ref, lap_no, ref_no, grid_dist, delta, ref_corners
    )
    figures["sections"] = plot_sections(sections, lap_no, ref_no)

    if ref_corners.empty:
        print("No corners found on the reference lap; "
              "skipping corner analysis.")
    else:
        corners = analyze_corners(
            lap, ref, lap_events, ref_corners, grid_dist, delta
        )
        print_corner_comparison(corners, lap_no, ref_no)
        print_coaching(corners, lap_no, ref_no)
        figures["corners"] = plot_corners(corners, lap_no, ref_no)

    can_map = (
        {"pos_x", "pos_z"}.issubset(ref.columns)
        and ref[["pos_x", "pos_z"]].notna().all().all()
        and ref[["pos_x", "pos_z"]].std().max() > 1
    )
    if can_map:
        figures["track_map"] = plot_track_map(
            ref, ref_corners, ref_no, mirror=not args.no_mirror_map
        )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            out = save_dir / f"{csv_path.stem}_lap{lap_no}_vs_{ref_no}_{name}.png"
            fig.savefig(out, dpi=150)
        print()
        print(f"Saved {len(figures)} plots to {save_dir}")

    if not args.no_show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)


if __name__ == "__main__":
    main()