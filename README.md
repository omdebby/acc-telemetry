# acc-telemetry

Python-based telemetry analyzer for Assetto Corsa Competitizone (ACC). Visually indicates speed in different sections of the track and compares lap times, brake and throttle inputs, and time deltas. Works best with Python 3.12, but other versions may also work.


To use:
Install both acc_reader.py and analyze.py. The file acc_reader.py is used to collect racing telemetry while driving and exports raw telemetry data to a .csv file. In addition, the analyzer also generates a JSON file containing metadata such as track, car, and recording settings. On the other hand, the file analyze.py takes in the most recent telemetry file and provides visual analysis through matplotlib in addition to verbal analysis.


For acc_reader.py:
Before running the script, make sure the following python modules are installed: argparse, csv, json, time, datetime, pathlib

Open Windows Powershell and enter a python virtual environment: .\.venv\Scripts\activate.ps1 (for Python 3.12). It's also helpful to make a folder for telemetry files, which can be accessed in Powershell through the cd command. Load into a session on ACC, then run the script with python acc_reader.py. The analyzer will not record data unless you are actively driving, and you can press Ctrl + C to stop recording telemetry. A .csv and a .json file will be saved to the directory you are in.


For analyze.py:
Before running the script, make sure the following python modules are installed: argparse, sys, dataclasses, pathlib, matplotlib, numpy, pandas

Follow the aforementioned instructions for entering a python virtual environment. Run the script with python analyze.py
