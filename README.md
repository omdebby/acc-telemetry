# ACC Telemetry desktop analyzer

## Start here

Double-click **Launch.cmd**, then **Open recording** and choose a telemetry CSV.
The launcher uses a .venv beside the app, your existing ACC-Telemetry environment,
or Python on PATH, in that order. It does not install anything automatically.

The app selects the fastest known-valid complete lap as the reference, and the
most recent other complete lap for comparison. You can change both selectors.
Use **Compare** after changing options.

Tabs include the lap inventory, speed/input/delta overview, track map (when
coordinates exist), sections, corner metrics, and an analysis report.
Each plot has zoom, pan, reset, and save controls. **Export results** creates a
new timestamped folder with PNG plots, CSV tables, and a text report.

## Corner numbering

The included **Barcelona ACC GP with final chicane** profile labels T1 through T16.
The original steering detector merged the two final right turns (T15 and T16).
Fixed track-position windows keep their identities stable even when the driver
changes steering input or takes a corner flat.

The profile loads automatically when the CSV's matching .meta.json identifies
Barcelona. Keep those files together. If the metadata is absent, select
Turn profile editor > Load profile > profiles/barcelona.json.

The turn sequence follows the older 16-turn layout, not the current 14-turn
F1 layout. Reference:
https://www.circuitcat.com/wp-content/uploads/2017/01/Manual-del-Oficial-de-Carrera-2017.pdf

Profile boundaries and label anchors were calibrated from the supplied October
recording, not surveyed official apex coordinates. They represent turn regions
and stable label positions. Do not use this profile for a different layout.

For other tracks, **Automatic estimates** detects turns from steering. An
expected count tunes the threshold but does NOT guarantee official numbering.
For dependable numbering:
1. Compare with automatic estimates.
2. In Turn profile editor, choose Seed from detected turns.
3. Edit entry, label/apex, and exit positions as percentages of the lap.
   Add or remove lines as needed. Lines become T1, T2, etc.
4. Apply edits, then Compare to inspect. Save the JSON profile for later use.

Profiles require the game's track_position data. Older files can still be
compared using estimated distance, but cannot use fixed track-position profiles.
Load a saved custom profile explicitly for each new recording.

## Setup on another Windows computer

Install Python 3.10 or newer with Tcl/Tk (included in the standard Windows
installer). In this folder run:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt

Then double-click Launch.cmd. Internet access is needed only for installation.
Analysis is local.

## Command-line analysis and recording

Existing command-line analysis remains available:

    python analyze.py recording.csv --no-show --save-dir plots
    python analyze.py recording.csv --lap 6 --reference 5
    python analyze.py recording.csv --profile profiles/barcelona.json
    python analyze.py recording.csv --auto-corners --corners 16

Recording remains a separate command-line utility. To install its dependency:

    python -m pip install -r requirements-recorder.txt
    python acc_reader.py --output-dir recordings

Drive in ACC and press Ctrl+C to stop recording. Record past the start/finish
crossing so the analyzer can confirm the previous lap was completed.
Live ACC capture has not been tested in a running game during this update.

## Corrections

- Stable Barcelona T1-T16 numbering, including separate T15/T16.
- Labels use turning locations instead of minimum-speed entry samples.
- Staggered labels and leader lines improve plot readability.
- Track progress is no longer stretched independently per lap.
- With track-position alignment, delta endpoints use zero at the start and the
  official lap time at the finish. This is an endpoint anchor, not an independent
  measurement of official lap time.
- Distance uses trapezoidal speed integration and remains an estimate in metres.
- Unfinished final laps require evidence of a subsequent lap crossing.
- Known-valid reference laps take priority over unknown-validity laps.
- Missing coordinates, empty corner detection, malformed numeric data, and
  invalid options are handled.
- Exit speed outside the recorded range is reported as unavailable.
- Recorder maps player ID through the car-ID array, captures packed_id, recognizes
  numeric LIVE status, strips null padding in metadata, validates intervals, and
  uses unique filenames.

## Interpretation and limits

Positive delta means the analyzed lap is slower. Positive braking-point difference
means braking later. Corner windows include entry/exit and can overlap; do not add
their time differences together. The section table partitions the full lap.

The recorded world coordinates in older CSVs cannot be repaired if the original
reader captured a different car. Re-record with the updated recorder in that case.

Recordings spanning a reset of the completed-lap counter are rejected with an
explanation; split these into individual sessions. Large telemetry gaps and
speed-integrated legacy comparisons can reduce accuracy.

Two files in the supplied folder can be analyzed: September 30 (legacy estimated
alignment) and October 6 at 16:19:32 (Barcelona profile). September 24 lacks lap
and steering columns; October 6 at 16:15:13 has no complete comparable laps.

## Validation

    python -m unittest -v test_telemetry

14 regression tests cover turn labels, profile validation, progress alignment,
official delta endpoints, section totals, incomplete laps, reference selection,
legacy input, malformed input, zero progress, player coordinates, and LIVE status.
The GUI was also exercised for comparison, plot creation, profile round trips,
export, automatic detection, and invalid lap selection.

The example-plots folder contains the corrected Barcelona comparison of lap 6
against lap 5. Official difference: +0.093 seconds.
