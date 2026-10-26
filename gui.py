"""Desktop telemetry analysis. Run: python gui.py [recording.csv]."""
import contextlib
import io
import json
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import analyze as a


class TelemetryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ACC Telemetry | Lap analysis")
        self.geometry("1280x900")
        self.minsize(1000, 720)
        self.path = None
        self.laps = []
        self.profile = None
        self.figures = {}
        self.result = None
        self.messages = queue.Queue()
        self.loading = False
        self.status = tk.StringVar(value="Open a telemetry CSV to begin.")
        self.lap_var = tk.StringVar()
        self.ref_var = tk.StringVar()
        self.mode = tk.StringVar(value="Automatic estimates")
        self.expected = tk.StringVar()
        self.mirror = tk.BooleanVar(value=True)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("Treeview", rowheight=27)
        header = ttk.Frame(self, padding=(18, 12))
        header.pack(fill="x")
        ttk.Label(header, text="ACC TELEMETRY", style="Title.TLabel").pack(side="left")
        self.open_btn = ttk.Button(header, text="Open recording", command=self.open_file)
        self.open_btn.pack(side="right")
        self.file_label = ttk.Label(self, text="Compare laps. Understand every turn.", padding=(18, 2))
        self.file_label.pack(fill="x")
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        for label, variable in (("Analyze lap", self.lap_var), ("Reference lap", self.ref_var)):
            ttk.Label(controls, text=label).pack(side="left", padx=(6, 5))
            box = ttk.Combobox(controls, textvariable=variable, state="readonly", width=10)
            box.pack(side="left")
            setattr(self, "lap_box" if variable is self.lap_var else "ref_box", box)
        ttk.Label(controls, text="Turns").pack(side="left", padx=(18, 5))
        self.mode_box = ttk.Combobox(controls, textvariable=self.mode, state="readonly", width=22,
                                    values=["Track profile", "Automatic estimates"])
        self.mode_box.pack(side="left")
        ttk.Label(controls, text="Expected count").pack(side="left", padx=(12, 5))
        ttk.Entry(controls, textvariable=self.expected, width=4).pack(side="left")
        self.compare_btn = ttk.Button(controls, text="Compare", command=self.compare, state="disabled")
        self.compare_btn.pack(side="left", padx=12)
        self.export_btn = ttk.Button(controls, text="Export results", command=self.export, state="disabled")
        self.export_btn.pack(side="right")
        self.summary = ttk.Label(self, text="", padding=(18, 4), font=("Segoe UI", 11, "bold"))
        self.summary.pack(fill="x")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)
        self.lap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.lap_frame, text="Recording laps")
        self.table = ttk.Treeview(self.lap_frame, columns=("lap", "time", "valid", "complete"),
                                 show="headings", selectmode="browse")
        for col, label in zip(self.table["columns"], ("Lap", "Time", "Validity", "Complete")):
            self.table.heading(col, text=label)
            self.table.column(col, width=180, anchor="center")
        self.table.pack(fill="both", expand=True)
        ttk.Label(self.lap_frame, text="Lap IDs match the CSV. Only complete laps can be compared. "
                  "Unknown validity is shown explicitly.", padding=10).pack(fill="x")
        self.plot_tabs = []
        self.editor_frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.editor_frame, text="Turn profile editor")
        ttk.Label(self.editor_frame, text="One turn per line: entry %, label/apex %, exit %. "
                  "Lines become T1, T2, ... in track order.").pack(anchor="w")
        ttk.Label(self.editor_frame, text="For tracks without a profile: compare with automatic estimates, "
                  "then seed this editor and correct the positions using the overview.").pack(anchor="w", pady=(3, 8))
        tools = ttk.Frame(self.editor_frame)
        tools.pack(fill="x")
        for label, fn in (("Load profile", self.load_profile), ("Seed from detected turns", self.seed_profile),
                          ("Apply edits", self.apply_profile), ("Save profile", self.save_profile)):
            ttk.Button(tools, text=label, command=fn).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(tools, text="Mirror map", variable=self.mirror).pack(side="right")
        self.editor = tk.Text(self.editor_frame, font=("Consolas", 12), undo=True, wrap="none")
        self.editor.pack(fill="both", expand=True, pady=10)
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=8).pack(fill="x")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.poll)
        if len(sys.argv) > 1:
            self.after(150, lambda: self.load_file(Path(sys.argv[1])))

    def open_file(self):
        path = filedialog.askopenfilename(title="Open telemetry", filetypes=[("Telemetry CSV", "*.csv")])
        if path:
            self.load_file(Path(path))

    def load_file(self, path):
        if self.loading:
            return
        self.loading = True
        self.open_btn.configure(state="disabled")
        self.compare_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status.set("Reading recording...")
        def worker():
            try:
                data = a.load_telemetry(path)
                laps = a.find_laps(data)
                warning = ""
                try:
                    profile = a.load_profile(path)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    profile, warning = None, f"Metadata/profile unavailable: {exc}"
                self.messages.put(("loaded", (path, data, laps, profile, warning)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            kind, payload = self.messages.get_nowait()
            self.loading = False
            self.open_btn.configure(state="normal")
            if kind == "error":
                self.status.set("Could not open recording.")
                messagebox.showerror("Open recording", payload)
                self.compare_btn.configure(state="normal" if self.laps else "disabled")
                self.export_btn.configure(state="normal" if self.result else "disabled")
            else:
                self.path, self.data, self.laps, self.profile, warning = payload
                self.clear_results()
                self.file_label.configure(text=str(self.path))
                self.table.delete(*self.table.get_children())
                for lap in self.laps:
                    self.table.insert("", "end", values=(lap.number, a.fmt_ms(lap.time_ms),
                                      {True:"Valid", False:"Invalid", None:"Unknown"}[lap.valid],
                                      "Yes" if lap.complete else "No"))
                values = [str(lap.number) for lap in self.laps if lap.complete]
                self.lap_box.configure(values=values)
                self.ref_box.configure(values=values)
                self.lap_var.set("")
                self.ref_var.set("")
                self.mode.set("Track profile" if self.profile else "Automatic estimates")
                self.fill_editor()
                self.expected.set("")
                if len(values) >= 2:
                    chosen, ref = a.choose_laps(self.laps, None, None)
                    self.lap_var.set(str(chosen.number))
                    self.ref_var.set(str(ref.number))
                    self.compare_btn.configure(state="normal")
                    self.compare()
                else:
                    self.status.set("At least two complete laps are needed. Keep recording past the finish line.")
                if warning:
                    self.status.set(warning)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def clear_results(self):
        for fig in self.figures.values():
            plt.close(fig)
        self.figures = {}
        self.result = None
        for tab in self.plot_tabs:
            self.notebook.forget(tab)
            tab.destroy()
        self.plot_tabs = []
        self.export_btn.configure(state="disabled")
        self.summary.configure(text="")

    def compare(self):
        if self.loading:
            return
        try:
            chosen, reference = a.choose_laps(self.laps, int(self.lap_var.get()), int(self.ref_var.get()))
            if chosen.number == reference.number:
                raise ValueError("Choose two different laps.")
            use_position = all("track_position" in info.data and info.data.track_position.notna().all()
                               for info in (chosen, reference))
            ref = a.build_lap(reference, use_position)
            length = ref.attrs["raw_length_m"]
            ref["dist_m"] = ref.progress * length
            lap = a.build_lap(chosen, use_position, length)
            ref_events, lap_events = a.detect_braking(ref), a.detect_braking(lap)
            if self.mode.get() == "Track profile":
                if self.profile is None:
                    raise ValueError("Load or create a turn profile, or select Automatic estimates.")
                corners = a.profile_corners(ref, ref_events, self.profile)
                source = self.profile.get("name", "Custom profile")
            elif self.expected.get().strip():
                corners, _ = a.tune_corners(ref, ref_events, int(self.expected.get()))
                source = f"Estimated turns: {len(corners)} (requested {self.expected.get()})"
            else:
                corners = a.detect_corners(ref, ref_events)
                source = "Estimated turns; official numbering is not guaranteed"
            grid = np.linspace(0, 1, a.GRID_POINTS)
            dist = grid * length
            delta = a.compute_delta(lap, ref, grid)
            sections = a.analyze_sections(lap, ref, 10)
            metrics = a.analyze_corners(lap, ref, lap_events, corners, dist, delta)
            self.clear_results()
            n, r = chosen.number, reference.number
            self.figures["overview"] = a.plot_overview(lap, ref, n, r, dist, delta, corners)
            if {"pos_x", "pos_z"}.issubset(ref.columns) and ref[["pos_x", "pos_z"]].notna().all().all() and ref[["pos_x", "pos_z"]].std().max() > 1:
                self.figures["track_map"] = a.plot_track_map(ref, corners, r, self.mirror.get())
            self.figures["sections"] = a.plot_sections(sections, n, r)
            if not metrics.empty:
                self.figures["corners"] = a.plot_corners(metrics, n, r)
            for name, fig in self.figures.items():
                tab = ttk.Frame(self.notebook)
                self.notebook.add(tab, text=name.replace("_", " ").title())
                self.plot_tabs.append(tab)
                fig.set_layout_engine("tight")
                canvas = FigureCanvasTkAgg(fig, master=tab)
                NavigationToolbar2Tk(canvas, tab).update()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                canvas.draw()
            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                print(f"Recording: {self.path.name}\n{source}\n")
                print(f"Lap {n}: {a.fmt_ms(chosen.time_ms)} | Reference {r}: {a.fmt_ms(reference.time_ms)}")
                print(f"Official difference: {(chosen.time_ms-reference.time_ms)/1000:+.3f} s")
                print("Corner windows can overlap; do not sum them. Section times partition the lap.")
                if not use_position:
                    print("Legacy distance alignment is estimated from speed.")
                if not metrics.empty:
                    a.print_corner_comparison(metrics, n, r)
                    a.print_coaching(metrics, n, r)
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text="Analysis report")
            self.plot_tabs.append(tab)
            text = tk.Text(tab, wrap="word", font=("Consolas", 10))
            scroll = ttk.Scrollbar(tab, command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)
            text.insert("1.0", report.getvalue())
            text.configure(state="disabled")
            self.result = dict(corners=corners, metrics=metrics, sections=sections, ref=ref,
                               report=report.getvalue(), lap=n, reference=r)
            self.summary.configure(text=f"Lap {n}  {a.fmt_ms(chosen.time_ms)}   |   Reference {r}  "
                                   f"{a.fmt_ms(reference.time_ms)}   |   Delta {(chosen.time_ms-reference.time_ms)/1000:+.3f} s")
            self.status.set(source + " | " + ("Track-position alignment" if use_position else "Estimated distance alignment"))
            self.export_btn.configure(state="normal")
            self.notebook.select(self.plot_tabs[0])
        except Exception as exc:
            messagebox.showerror("Compare laps", str(exc))

    def fill_editor(self):
        self.editor.delete("1.0", "end")
        if self.profile:
            self.editor.insert("1.0", "\n".join(
                " ".join(f"{float(t[k])*100:.4f}" for k in ("start", "apex", "end"))
                for t in self.profile["turns"]))

    def editor_profile(self):
        turns = []
        for line in self.editor.get("1.0", "end").splitlines():
            if not line.strip():
                continue
            values = [float(v)/100 for v in line.replace(",", " ").split()]
            if len(values) != 3:
                raise ValueError("Each line needs exactly three percentages: entry, label/apex, exit.")
            turns.append(dict(zip(("start", "apex", "end"), values)))
        return a.validate_profile(dict(name="Custom turn profile", turns=turns))

    def apply_profile(self):
        try:
            self.profile = self.editor_profile()
            self.mode.set("Track profile")
            self.status.set("Turn profile applied. Click Compare to update plots.")
        except Exception as exc:
            messagebox.showerror("Turn profile", str(exc))

    def load_profile(self):
        path = filedialog.askopenfilename(filetypes=[("Turn profile", "*.json")])
        if path:
            try:
                self.profile = a.load_profile(self.path or "", path)
                self.fill_editor()
                self.mode.set("Track profile")
                self.status.set("Profile loaded. Click Compare to update plots.")
            except Exception as exc:
                messagebox.showerror("Turn profile", str(exc))

    def seed_profile(self):
        if not self.result:
            return
        if not self.result["ref"].attrs.get("positioned"):
            messagebox.showinfo("Turn profile", "Profiles require track_position data; this recording uses estimated distance.")
            return
        length = self.result["ref"].attrs["raw_length_m"]
        self.profile = dict(name="Custom turn profile", turns=[
            dict(start=row.start_dist/length, apex=row.apex_dist/length, end=row.end_dist/length)
            for row in self.result["corners"].itertuples()])
        self.fill_editor()
        self.status.set("Review and correct the detected windows, then Apply edits.")

    def save_profile(self):
        try:
            profile = self.editor_profile()
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Turn profile", "*.json")])
            if path:
                Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")
                self.status.set(f"Saved turn profile: {path}")
        except Exception as exc:
            messagebox.showerror("Save profile", str(exc))

    def export(self):
        if not self.result:
            return
        folder = filedialog.askdirectory(title="Export plots and tables")
        if not folder:
            return
        try:
            from datetime import datetime
            dest = Path(folder) / (f"{self.path.stem}_lap{self.result['lap']}_vs_{self.result['reference']}_"
                                   + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            dest.mkdir()
            for name, fig in self.figures.items():
                fig.savefig(dest/f"{name}.png", dpi=150, bbox_inches="tight")
            for name in ("corners", "metrics", "sections"):
                self.result[name].to_csv(dest/f"{name}.csv", index=False)
            (dest/"report.txt").write_text(self.result["report"], encoding="utf-8")
            self.status.set(f"Exported to {dest}")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def close(self):
        for fig in self.figures.values():
            plt.close(fig)
        self.destroy()


if __name__ == "__main__":
    TelemetryApp().mainloop()
