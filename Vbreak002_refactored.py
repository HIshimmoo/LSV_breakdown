#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored LSV Overpotential Analysis GUI
- Cleaner ttk UI with sensible layout and sizes
- Removed duplicate frames and noisy comments
- Fixed export sheet labels to reflect "stacked" curves
- Kept original math and fitting logic; consolidated helpers
"""

import os
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import linregress

# ---------------------------
# Constants
# ---------------------------
WINDOW_WIDTH = 0.06    # 60 mV candidate window for Tafel fitting
GRID_STEP = 0.005      # Voltage step used in grid search

# ---------------------------
# I/O
# ---------------------------
def read_data(filename):
    """
    Read first two columns (Voltage, Current) from .txt (whitespace-delimited, no header)
    or .xlsx (first sheet). Filters to positive current and drops NaNs.
    Returns (V, i) as NumPy arrays, or (None, None) on error.
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".xlsx":
            data = pd.read_excel(filename)
        elif ext == ".txt":
            data = pd.read_csv(filename, delim_whitespace=True, header=None)
        else:
            messagebox.showerror("File Error", "Only .txt and .xlsx files are supported.")
            return None, None
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading the file:\n{e}")
        return None, None

    if data.shape[1] < 2:
        messagebox.showerror("Column Error", "The file must have at least two columns (Voltage, Current).")
        return None, None

    V = data.iloc[:, 0]
    i = data.iloc[:, 1]
    mask = (i > 0) & V.notnull() & i.notnull()
    V = V[mask].to_numpy()
    i = i[mask].to_numpy()

    if len(V) == 0:
        messagebox.showerror("Data Error", "No valid data found (need positive currents and no NaNs).")
        return None, None

    return V, i

# ---------------------------
# Tafel fitting core
# ---------------------------
def fit_tafel_window(eta_corr, i, i_lower=0.005, i_upper=0.1,
                     window_width=WINDOW_WIDTH, step=GRID_STEP):
    """
    Grid-search a 60-mV candidate window; test 20/40/60 mV subwindows.
    Choose window minimizing combined relative errors of slope and i0.
    """
    mask_range = (i >= i_lower) & (i <= i_upper)
    if np.sum(mask_range) < 5:
        messagebox.showerror(
            "Data Error",
            f"Not enough points in the chosen range: {i_lower:.4g}–{i_upper:.4g} A/cm²."
        )
        return None

    eta_sel = eta_corr[mask_range]
    i_sel = i[mask_range]
    vmin, vmax = np.min(eta_sel), np.max(eta_sel)
    if vmax - vmin < window_width:
        messagebox.showerror("Data Error", "Voltage span too narrow for a 60 mV window.")
        return None

    candidate_results = []
    for start in np.arange(vmin, vmax - window_width, step):
        end = start + window_width
        cand_mask = (eta_sel >= start) & (eta_sel <= end)
        if np.sum(cand_mask) < 3:
            continue

        center = 0.5 * (start + end)
        subwindows = {
            "20mV": (center - 0.01, center + 0.01),
            "40mV": (center - 0.02, center + 0.02),
            "60mV": (start, end),
        }

        slopes, intercepts, r2s = [], [], []
        counts = {}
        for key, (low, high) in subwindows.items():
            sub_mask = (eta_sel >= low) & (eta_sel <= high)
            n = int(np.sum(sub_mask))
            counts[key] = n
            if n < 3:
                slopes.append(np.nan); intercepts.append(np.nan); r2s.append(np.nan)
                continue
            i_sub = i_sel[sub_mask]
            eta_sub = eta_sel[sub_mask]
            res = linregress(np.log10(i_sub), eta_sub)
            slopes.append(res.slope)
            intercepts.append(res.intercept)
            r2s.append(res.rvalue ** 2)

        slopes = np.array(slopes)
        if np.any(np.isnan(slopes)):
            continue

        i0s = np.array([10 ** (-intercepts[j] / slopes[j]) for j in range(len(slopes))])
        avg_slope = float(np.mean(slopes))
        std_slope = float(np.std(slopes))
        avg_intercept = float(np.mean(intercepts))
        avg_i0 = float(np.mean(i0s))
        std_i0 = float(np.std(i0s))

        rel_err_b = std_slope / abs(avg_slope) if avg_slope != 0 else np.inf
        rel_err_i0 = std_i0 / avg_i0 if avg_i0 != 0 else np.inf
        metric = 0.5 * (rel_err_b + rel_err_i0)

        candidate_results.append({
            "window": (start, end),
            "center": center,
            "subwindows": subwindows,
            "slopes": slopes,
            "avg_slope": avg_slope,
            "std_slope": std_slope,
            "intercepts": intercepts,
            "avg_intercept": avg_intercept,
            "i0s": i0s,
            "avg_i0": avg_i0,
            "std_i0": std_i0,
            "rel_err_b": rel_err_b,
            "rel_err_i0": rel_err_i0,
            "candidate_metric": metric,
            "avg_r2": float(np.mean(r2s)),
            "counts": counts,
            "n_points": int(np.sum(mask_range))
        })

    if not candidate_results:
        messagebox.showerror("Fitting Error", "No valid fits in any candidate window.")
        return None

    best = min(candidate_results, key=lambda x: x["candidate_metric"])
    return {
        "b_kin": best["avg_slope"],
        "intercept": best["avg_intercept"],
        "i0": 10 ** (-best["avg_intercept"] / best["avg_slope"]),
        "r_squared": best["avg_r2"],
        "n_points": best["n_points"],
        "candidate_details": best
    }

def calculate_RCL_overpotential(i, R_CL, b_kin):
    """
    Catalyst-layer overpotential from Tafel slope and R_CL via utilization U_J.
    """
    term = (i * np.log(10) * R_CL) / (2 * b_kin)
    term = np.clip(term, 0, None)
    U_J = (1 + term ** 1.1982) ** (-1 / 1.1982)
    return -b_kin * np.log10(U_J)

def iterative_tafel_fit(V, i, E_rev, HFR, R_CL,
                        i_lower=0.005, i_upper=0.1,
                        tol=1e-3, max_iter=10):
    """
    Iteratively fit Tafel slope while subtracting ohmic and R_CL drops.
    Stops when slope converges and accumulated overpotential at i_upper
    does not exceed V - E_rev.
    """
    b_prev = None
    fit_result = None

    for _ in range(max_iter):
        eta_ohm = i * HFR
        eta_rcl = np.zeros_like(i) if b_prev is None else calculate_RCL_overpotential(i, R_CL, b_prev)
        eta_corr = (V - E_rev) - eta_ohm - eta_rcl

        fit_result = fit_tafel_window(eta_corr, i, i_lower, i_upper)
        if fit_result is None:
            return None

        b_new = fit_result["b_kin"]
        i0 = fit_result["i0"]

        eta_kin_all = b_new * np.log10(i / i0)
        eta_ohm_all = i * HFR
        eta_rcl_all = calculate_RCL_overpotential(i, R_CL, b_new)
        acc_all = eta_kin_all + eta_ohm_all + eta_rcl_all

        idxs = np.where(i >= i_upper)[0]
        condition_ok = True
        if len(idxs) > 0:
            idx = idxs[0]
            allowed = V[idx] - E_rev
            condition_ok = (acc_all[idx] <= allowed)

        if (b_prev is not None) and (abs(b_new - b_prev) < tol) and condition_ok:
            return fit_result, eta_ohm_all, eta_rcl_all, acc_all

        b_prev = b_new

    messagebox.showwarning("Iteration Warning", "Tafel fitting did not fully converge.")
    return fit_result, eta_ohm_all, eta_rcl_all, acc_all

# ---------------------------
# GUI
# ---------------------------
class IntegratedLSVAnalysisApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Integrated LSV Overpotential Analysis")
        self._init_dpi_awareness()
        self._init_style()

        # Window geometry (responsive)
        sw, sh = master.winfo_screenwidth(), master.winfo_screenheight()
        master.geometry(f"{int(sw * 0.9)}x{int(sh * 0.9)}")
        master.minsize(1100, 700)

        self.pack(fill=tk.BOTH, expand=True)

        # --- Data holders ---
        self.filename = None
        self.V_data = None
        self.i_data = None
        self.E_rev = None
        self.fit_results = None
        self.eta_kin = None
        self.Eta_ohm = None
        self.Eta_RCL = None
        self.Eta_res = None
        self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = self.y6 = None
        self.x_log_scale = False

        # Reference electrodes
        self.reference_options = {
            "RHE": 0.0,
            "Ag/AgCl (sat)": 0.197,
            "SCE (sat)": 0.242,
            "HgO (sat)": 0.098
        }

        # --- Layout ---
        self._build_top_controls()
        self._build_diag()
        self._build_plot()
        self._build_statusbar()

    # ----- UI helpers -----
    def _init_dpi_awareness(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    def _init_style(self):
        style = ttk.Style()
        # Prefer platform theme; fall back to 'clam'
        theme = style.theme_use()
        if theme not in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", padding=6)
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TCombobox", font=("Segoe UI", 10))

    def _build_top_controls(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        # --- Row 1: Data ---
        box_data = ttk.LabelFrame(top, text="Data")
        box_data.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(box_data, textvariable=self.file_var).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Button(box_data, text="Browse...", command=self.choose_file).grid(row=0, column=1, sticky="e", padx=4, pady=4)
        ttk.Label(box_data, text="Allowed: .txt or .xlsx").grid(row=0, column=2, sticky="w", padx=4, pady=4)

        # --- Row 2: Electrolyte & Reference ---
        box_ref = ttk.LabelFrame(top, text="Electrolyte & Reference")
        box_ref.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Label(box_ref, text="Reference Electrode").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.ref_var = tk.StringVar(value="RHE")
        self.ref_combo = ttk.Combobox(box_ref, textvariable=self.ref_var, values=list(self.reference_options.keys()), state="readonly", width=18)
        self.ref_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(box_ref, text="pH").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.pH_entry = ttk.Entry(box_ref, width=8)
        self.pH_entry.insert(0, "0")
        self.pH_entry.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(box_ref, text="Temperature (K)").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        self.temp_entry = ttk.Entry(box_ref, width=10)
        self.temp_entry.insert(0, "378")
        self.temp_entry.grid(row=0, column=5, sticky="w", padx=4, pady=4)

        # --- Row 3: Resistances ---
        box_R = ttk.LabelFrame(top, text="Resistances")
        box_R.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Label(box_R, text="R_CL (Ω·cm²)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.rcl_entry = ttk.Entry(box_R, width=10)
        self.rcl_entry.insert(0, "0")
        self.rcl_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(box_R, text="HFR (Ω·cm²)").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.hfr_entry = ttk.Entry(box_R, width=10)
        self.hfr_entry.insert(0, "0.01")
        self.hfr_entry.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        # --- Row 4: Tafel Fit ---
        box_tafel = ttk.LabelFrame(top, text="Tafel Fit Range")
        box_tafel.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Label(box_tafel, text="Lower (A/cm²)").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.tafel_lower_entry = ttk.Entry(box_tafel, width=10)
        self.tafel_lower_entry.insert(0, "0.02")
        self.tafel_lower_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(box_tafel, text="Upper (A/cm²)").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.tafel_upper_entry = ttk.Entry(box_tafel, width=10)
        self.tafel_upper_entry.insert(0, "0.8")
        self.tafel_upper_entry.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        # Buttons
        btns = ttk.Frame(top)
        btns.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        ttk.Button(btns, text="Fit & Plot", command=self.perform_fit).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Toggle X-Axis Scale", command=self.toggle_x_scale).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=4)

        # Grid weights for label frames
        for frame in (box_data, box_ref, box_R, box_tafel):
            for c in range(6):
                frame.grid_columnconfigure(c, weight=1)

    def _build_diag(self):
        box = ttk.LabelFrame(self, text="Fitting Details")
        box.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        self.diag_text = tk.Text(box, height=10, font=("Consolas", 10))
        self.diag_text.pack(fill=tk.BOTH, expand=True)
        self._write_diag("No details yet.")

    def _build_plot(self):
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Current Density (A/cm²)")
        self.ax.set_ylabel("Potential (V)")
        self.ax.set_title("LSV Overpotential Analysis")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready")
        bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _write_diag(self, text):
        self.diag_text.config(state=tk.NORMAL)
        self.diag_text.delete("1.0", tk.END)
        self.diag_text.insert(tk.END, text)
        self.diag_text.config(state=tk.DISABLED)

    # ----- Actions -----
    def choose_file(self):
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt")]
        )
        if filename:
            self.filename = filename
            self.file_var.set(filename)
            self.status_var.set(f"Selected file: {os.path.basename(filename)}")
        else:
            self.filename = None
            self.file_var.set("No file selected")
            self.status_var.set("Ready")

    def perform_fit(self):
        if not self.filename:
            messagebox.showerror("File Error", "No file selected.")
            return
        try:
            T = float(self.temp_entry.get())
            R_CL = float(self.rcl_entry.get())
            HFR = float(self.hfr_entry.get())
            pH_val = float(self.pH_entry.get())
            i_lower = float(self.tafel_lower_entry.get())
            i_upper = float(self.tafel_upper_entry.get())
            if i_lower >= i_upper:
                messagebox.showerror("Input Error", "Lower limit must be < Upper limit.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter numeric values for all parameters.")
            return

        # Thermodynamic potential (simple linear T dependence)
        self.E_rev = 1.2291 - 0.0008456 * (T - 298.15)

        V_raw, i_raw = read_data(self.filename)
        if V_raw is None:
            return

        # Reference conversion
        ref_choice = self.ref_var.get()
        offset = self.reference_options[ref_choice]
        if ref_choice == "RHE":
            V_corr = V_raw
        else:
            V_corr = V_raw + 0.0592 * pH_val + offset

        self.V_data = V_corr
        self.i_data = i_raw

        # Fit
        iterative_result = iterative_tafel_fit(
            self.V_data, self.i_data, self.E_rev, HFR, R_CL, i_lower, i_upper
        )
        if iterative_result is None:
            return

        fit_result, eta_ohm_all, eta_rcl_all, _ = iterative_result
        b_kin = fit_result["b_kin"]
        i0 = fit_result["i0"]
        self.fit_results = fit_result

        # Components
        self.eta_kin = b_kin * np.log10(self.i_data / i0)
        self.Eta_ohm = eta_ohm_all
        self.Eta_RCL = eta_rcl_all
        self.Eta_res = (self.V_data - self.E_rev) - (self.eta_kin + self.Eta_ohm + self.Eta_RCL)
        self.Eta_res = np.maximum(self.Eta_res, 0)

        # Stacked curves
        self.y1 = np.full_like(self.i_data, self.E_rev)              # E_rev
        self.y2 = self.y1 + self.eta_kin                             # + η_kin
        self.y3 = self.y2 + self.Eta_ohm                             # + η_ohm
        self.y4 = self.y3 + self.Eta_RCL                             # + η_RCL
        self.y5 = self.y4 + self.Eta_res                             # + η_res
        self.y6 = self.V_data                                        # original LSV

        # Diagnostics
        d = fit_result["candidate_details"]
        diag = (
            f"Tafel fit (iterative)\n"
            f"  b_kin: {b_kin:.4f} V/dec\n"
            f"  i0:    {i0:.4e} A/cm²\n"
            f"  Intercept: {fit_result['intercept']:.4f} V\n"
            f"  R²:    {fit_result['r_squared']:.4f}\n"
            f"  N in [{i_lower:.4g}, {i_upper:.4g}] A/cm²: {fit_result['n_points']}\n\n"
            f"Best 60 mV window\n"
            f"  {d['window'][0]:.4f}–{d['window'][1]:.4f} V  (center {d['center']:.4f} V)\n"
            f"  20 mV: {d['subwindows']['20mV'][0]:.4f}–{d['subwindows']['20mV'][1]:.4f} (n={d['counts']['20mV']})\n"
            f"  40 mV: {d['subwindows']['40mV'][0]:.4f}–{d['subwindows']['40mV'][1]:.4f} (n={d['counts']['40mV']})\n"
            f"  60 mV: {d['subwindows']['60mV'][0]:.4f}–{d['subwindows']['60mV'][1]:.4f} (n={d['counts']['60mV']})\n"
            f"  Rel. SE(b):  {d['rel_err_b']*100:.2f}%   Rel. SE(i0): {d['rel_err_i0']*100:.2f}%\n"
            f"  Combined metric: {d['candidate_metric']*100:.2f}%\n"
        )
        self._write_diag(diag)

        # Plot
        self._plot_curves()
        self.status_var.set("Fit complete.")

    def _plot_curves(self):
        tailwind_colors = [
            "#111827",  # Slate (E_rev baseline)
            "#2563EB",  # Blue (kinetic)
            "#059669",  # Green (ohmic)
            "#F59E0B",  # Amber (RCL)
            "#DC2626",  # Red (residual)
            "#6B7280",  # Gray (original)
        ]

        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xscale("log" if self.x_log_scale else "linear")

        l1, = self.ax.plot(self.i_data, self.y1, color=tailwind_colors[0],
                           linestyle="--", linewidth=2, label="E_rev")
        l2, = self.ax.plot(self.i_data, self.y2, color=tailwind_colors[1],
                           linewidth=2, label="E_rev + η_kin")
        l3, = self.ax.plot(self.i_data, self.y3, color=tailwind_colors[2],
                           linewidth=2, label="+ η_ohm")
        l4, = self.ax.plot(self.i_data, self.y4, color=tailwind_colors[3],
                           linewidth=2, label="+ η_RCL")
        l5, = self.ax.plot(self.i_data, self.y5, color=tailwind_colors[4],
                           linewidth=2, label="+ η_res")
        l6, = self.ax.plot(self.i_data, self.y6, color=tailwind_colors[5],
                           marker="o", linestyle="none", markersize=4, label="Original LSV (RHE)")

        self.lines = {
            "E_rev": l1, "E_rev_kin": l2, "E_rev_kin_ohm": l3,
            "E_rev_kin_ohm_rcl": l4, "E_rev_kin_ohm_rcl_res": l5, "Original": l6
        }

        self.ax.set_xlabel("Current Density (A/cm²)")
        self.ax.set_ylabel("Potential (V)")
        self.ax.set_title("LSV Overpotential Analysis")
        self.ax.legend(fontsize=9, ncol=2)
        self.canvas.draw()

    def export_data(self):
        if self.i_data is None or self.y1 is None:
            messagebox.showerror("Export Error", "No data available. Perform a fit first.")
            return

        # Sheet 1: Stacked/overlay curves for plotting
        df_curves = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "E_rev": self.y1,
            "E_rev + η_kin": self.y2,
            "+ η_ohm": self.y3,
            "+ η_RCL": self.y4,
            "+ η_res": self.y5,
            "Original LSV (RHE)": self.y6
        })

        # Sheet 2: Overpotential components only
        df_components = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "η_kin (V)": self.eta_kin,
            "η_ohm (V)": self.Eta_ohm,
            "η_RCL (V)": self.Eta_RCL,
            "η_res (V)": self.Eta_res
        })

        # Sheet 3: Fit info
        if self.fit_results is not None:
            df_fit_info = pd.DataFrame({
                "Parameter": ["Tafel slope (V/dec)", "Exchange current density (A/cm²)",
                              "Intercept (V)", "R²", "N points"],
                "Value": [self.fit_results["b_kin"],
                          self.fit_results["i0"],
                          self.fit_results["intercept"],
                          self.fit_results["r_squared"],
                          self.fit_results["n_points"]]
            })
        else:
            df_fit_info = pd.DataFrame()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Data As"
        )
        if not file_path:
            return
        try:
            with pd.ExcelWriter(file_path) as writer:
                df_curves.to_excel(writer, sheet_name="Curves (stacked)", index=False)
                df_components.to_excel(writer, sheet_name="Overpotential Components", index=False)
                df_fit_info.to_excel(writer, sheet_name="Fitting Info", index=False)
            messagebox.showinfo("Export Successful", f"Data exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data:\n{e}")

    def toggle_x_scale(self):
        if self.i_data is None or self.y1 is None:
            messagebox.showerror("Plot Error", "No plot available. Perform a fit first.")
            return
        self.x_log_scale = not self.x_log_scale
        self._plot_curves()
        self.status_var.set(f"X-axis scale: {'log' if self.x_log_scale else 'linear'}")

# ---------------------------
# Main
# ---------------------------
def main():
    root = tk.Tk()
    app = IntegratedLSVAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
