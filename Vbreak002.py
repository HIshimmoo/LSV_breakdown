import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import linregress
import os

# --- Global Constants ---
WINDOW_WIDTH = 0.06    # 60 mV candidate window for Tafel fitting
GRID_STEP = 0.005      # Voltage step used in grid search

# ---------------------------
# Data Reading Function
# ---------------------------
def read_data(filename):
    """
    Reads the first two columns from a .txt or .xlsx file.
    Returns (V, i) as NumPy arrays or (None, None) on error.
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".xlsx":
            data = pd.read_excel(filename)
        elif ext == ".txt":
            # Assumes whitespace-delimited file with no header.
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

    # Keep only rows with positive current and no NaNs.
    V = data.iloc[:, 0]
    i = data.iloc[:, 1]
    mask = (i > 0) & V.notnull() & i.notnull()
    V = V[mask].to_numpy()
    i = i[mask].to_numpy()

    if len(V) == 0:
        messagebox.showerror("Data Error", "No valid data found (check for positive currents, no NaN).")
        return None, None

    return V, i

# ---------------------------
# Candidate Tafel Fitting (Grid Search)
# ---------------------------
def fit_tafel_window(eta_corr, i, i_lower=0.005, i_upper=0.1, window_width=WINDOW_WIDTH, step=GRID_STEP):
    """
    Performs a systematic grid search for a candidate voltage window (of window_width) that
    provides a stable Tafel fit over three subwindows (20, 40, 60 mV). Returns a dictionary
    with the best candidate (lowest combined relative error of the slope and i0).
    """
    mask_range = (i >= i_lower) & (i <= i_upper)
    if np.sum(mask_range) < 5:
        messagebox.showerror(
            "Data Error",
            f"Not enough data points in the chosen current-density range: {i_lower:.4g}–{i_upper:.4g} A/cm²."
        )
        return None

    eta_sel = eta_corr[mask_range]
    i_sel = i[mask_range]
    vmin, vmax = np.min(eta_sel), np.max(eta_sel)
    if vmax - vmin < window_width:
        messagebox.showerror("Data Error", "The voltage range in the selected current region is too narrow for a 60 mV window.")
        return None

    candidate_results = []
    candidate_starts = np.arange(vmin, vmax - window_width, step)
    for start in candidate_starts:
        end = start + window_width
        candidate_mask = (eta_sel >= start) & (eta_sel <= end)
        if np.sum(candidate_mask) < 3:
            continue
        center = (start + end) / 2.0
        subwindows = {
            '20mV': (center - 0.01, center + 0.01),
            '40mV': (center - 0.02, center + 0.02),
            '60mV': (start, end)
        }
        slopes = []
        intercepts = []
        r2s = []
        counts = {}
        for key, (low, high) in subwindows.items():
            sub_mask = (eta_sel >= low) & (eta_sel <= high)
            counts[key] = np.sum(sub_mask)
            if np.sum(sub_mask) < 3:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                r2s.append(np.nan)
            else:
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
        avg_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        avg_i0 = np.mean(i0s)
        std_i0 = np.std(i0s)
        rel_err_b = std_slope / abs(avg_slope) if avg_slope != 0 else np.inf
        rel_err_i0 = std_i0 / avg_i0 if avg_i0 != 0 else np.inf
        candidate_metric = (rel_err_b + rel_err_i0) / 2.0
        avg_r2 = np.mean(r2s)

        candidate_results.append({
            "window": (start, end),
            "center": center,
            "subwindows": subwindows,
            "slopes": slopes,
            "avg_slope": avg_slope,
            "std_slope": std_slope,
            "intercepts": intercepts,
            "avg_intercept": np.mean(intercepts),
            "i0s": i0s,
            "avg_i0": avg_i0,
            "std_i0": std_i0,
            "rel_err_b": rel_err_b,
            "rel_err_i0": rel_err_i0,
            "candidate_metric": candidate_metric,
            "avg_r2": avg_r2,
            "counts": counts,
            "n_points": np.sum(mask_range)
        })

    if not candidate_results:
        messagebox.showerror("Fitting Error", "Could not obtain valid fits in any candidate window.")
        return None

    best_candidate = min(candidate_results, key=lambda x: x["candidate_metric"])
    result = {
        "b_kin": best_candidate["avg_slope"],
        "intercept": best_candidate["avg_intercept"],
        "i0": 10 ** (-best_candidate["avg_intercept"] / best_candidate["avg_slope"]),
        "r_squared": best_candidate["avg_r2"],
        "n_points": best_candidate["n_points"],
        "candidate_details": best_candidate
    }
    return result

# ---------------------------
# Iterative Tafel Fitting (with R_CL removal and accumulation condition)
# ---------------------------
def iterative_tafel_fit(V, i, E_rev, HFR, R_CL, i_lower=0.005, i_upper=0.1, tol=1e-3, max_iter=10):
    """
    Iteratively fits the Tafel slope while removing the influence of the catalyst-layer (R_CL)
    overpotential. In the first iteration only the ohmic (HFR) drop is subtracted.
    In further iterations the estimated R_CL overpotential (based on the previous b_kin) is subtracted.
    The iteration will only stop when both the Tafel slope converges AND the predicted accumulated
    overpotential (η_kin + η_ohm + η_RCL) at the upper fit limit (i.e. the first data point with i >= i_upper)
    is less than (V – E_rev). If not, the iteration continues.
    Returns the fit result along with the arrays for ohmic, R_CL, and the accumulated overpotential.
    """
    b_kin_prev = None
    fit_result = None
    for iteration in range(max_iter):
        # Compute ohmic drop (always)
        Eta_ohm = i * HFR

        # For the first iteration, assume no R_CL drop
        if b_kin_prev is None:
            Eta_RCL = np.zeros_like(i)
        else:
            Eta_RCL = calculate_RCL_overpotential(i, R_CL, b_kin_prev)

        # Corrected overpotential for fitting:
        eta_corr = (V - E_rev) - Eta_ohm - Eta_RCL

        # Fit Tafel parameters using points in the range [i_lower, i_upper]
        fit_result = fit_tafel_window(eta_corr, i, i_lower, i_upper)
        if fit_result is None:
            return None
        b_kin_new = fit_result["b_kin"]
        i0 = fit_result["i0"]

        # Compute the predicted kinetic overpotential over the full dataset.
        eta_kin_all = b_kin_new * np.log10(i / i0)
        eta_ohm_all = i * HFR
        eta_RCL_all = calculate_RCL_overpotential(i, R_CL, b_kin_new)
        accumulation_all = eta_kin_all + eta_ohm_all + eta_RCL_all

        # Check the condition at the upper fit limit (first index with i >= i_upper)
        indices = np.where(i >= i_upper)[0]
        condition_met = True
        if len(indices) > 0:
            idx_target = indices[0]
            allowed = V[idx_target] - E_rev
            if accumulation_all[idx_target] > allowed:
                condition_met = False

        # If b_kin has converged AND the accumulation condition is met, end iteration.
        if (b_kin_prev is not None and abs(b_kin_new - b_kin_prev) < tol and condition_met):
            return fit_result, eta_ohm_all, eta_RCL_all, accumulation_all

        b_kin_prev = b_kin_new

    messagebox.showwarning("Iteration Warning", "Tafel fitting did not converge within the maximum iterations.")
    return fit_result, eta_ohm_all, eta_RCL_all, accumulation_all

def calculate_RCL_overpotential(i, R_CL, b_kin):
    """
    Calculates the catalyst-layer (R_CL) overpotential based on the Tafel slope (b_kin)
    and the current density array i.
    """
    term = (i * np.log(10) * R_CL) / (2 * b_kin)
    term = np.clip(term, 0, None)
    U_J = (1 + term ** 1.1982) ** (-1 / 1.1982)
    return -b_kin * np.log10(U_J)

def calculate_i0(b_kin, intercept):
    return 10 ** (-intercept / b_kin)

# ---------------------------
# Integrated GUI Application
# ---------------------------
class IntegratedLSVAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Integrated LSV Overpotential Analysis")
        master.geometry("1400x900")
        self.font = ("Arial", 24)

        # Reference electrode offsets
        self.reference_options = {
            "RHE": 0.0,
            "Ag/AgCl (sat)": 0.197,
            "SCE (sat)": 0.242,
            "HgO (sat)": 0.098
        }

        # Top Frame: Input Parameters
        input_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 1) File browsing
        tk.Label(input_frame, text="Selected File:", font=self.font).grid(row=0, column=0, sticky="w")
        self.file_label = tk.Label(input_frame, text="None", font=self.font)
        self.file_label.grid(row=0, column=1, sticky="w", padx=5)
        self.browse_button = tk.Button(input_frame, text="Browse", font=self.font,
                                       command=self.choose_file, bg="#e6f2ff")
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        tk.Label(input_frame, text="(Allowed: .txt or .xlsx)", font=self.font, fg="blue").grid(row=0, column=3, sticky="w")

        # 2) Reference Electrode and pH
        tk.Label(input_frame, text="Reference Electrode:", font=self.font).grid(row=1, column=0, sticky="w", pady=5)
        self.ref_var = tk.StringVar(value="RHE")
        self.ref_menu = tk.OptionMenu(input_frame, self.ref_var, *self.reference_options.keys())
        self.ref_menu.config(font=self.font, width=15, bg="#ffffcc")
        self.ref_menu.grid(row=1, column=1, padx=5, sticky="w")
        menu = self.ref_menu["menu"]
        menu.config(font=("Arial", 24))
        tk.Label(input_frame, text="pH:", font=self.font).grid(row=1, column=2, sticky="w", pady=5)
        self.pH_entry = tk.Entry(input_frame, font=self.font, width=6)
        self.pH_entry.insert(0, "0")
        self.pH_entry.grid(row=1, column=3, padx=5, sticky="w")

        # 3) Temperature
        tk.Label(input_frame, text="Temperature (K):", font=self.font).grid(row=2, column=0, sticky="w")
        self.temp_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.temp_entry.insert(0, "298.15")
        self.temp_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 4) R_CL and 5) HFR
        tk.Label(input_frame, text="R_CL (Ohm·cm²):", font=self.font).grid(row=3, column=0, sticky="w")
        self.rcl_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.rcl_entry.insert(0, "0.1")
        self.rcl_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        tk.Label(input_frame, text="HFR (Ohm·cm²):", font=self.font).grid(row=3, column=2, sticky="w")
        self.hfr_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.hfr_entry.insert(0, "0.01")
        self.hfr_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # 6) Fit & Plot and 7) Export buttons
        self.fit_button = tk.Button(input_frame, text="Fit & Plot", font=self.font,
                                    command=self.perform_fit, bg="#cce6ff")
        self.fit_button.grid(row=4, column=0, columnspan=2, pady=10)
        self.export_button = tk.Button(input_frame, text="Export Data", font=self.font,
                                       command=self.export_data, bg="#d9ead3")
        self.export_button.grid(row=4, column=2, columnspan=2, pady=10)

        # 8) Toggle X-axis scale button
        self.x_log_scale = False
        self.toggle_xscale_button = tk.Button(input_frame, text="Toggle X-Axis Scale", font=self.font,
                                              command=self.toggle_x_scale, bg="#ffe699")
        self.toggle_xscale_button.grid(row=5, column=0, columnspan=4, pady=5)

        # Tafel Fit Range inputs
        tk.Label(input_frame, text="Tafel Fit Range:", font=self.font).grid(row=6, column=0, sticky="w", pady=5)
        tk.Label(input_frame, text="Lower (A/cm²):", font=self.font).grid(row=6, column=1, sticky="e")
        self.tafel_lower_entry = tk.Entry(input_frame, font=self.font, width=8)
        self.tafel_lower_entry.insert(0, "0.005")
        self.tafel_lower_entry.grid(row=6, column=2, padx=5, pady=5, sticky="w")
        tk.Label(input_frame, text="Upper (A/cm²):", font=self.font).grid(row=6, column=3, sticky="e")
        self.tafel_upper_entry = tk.Entry(input_frame, font=self.font, width=8)
        self.tafel_upper_entry.insert(0, "0.1")
        self.tafel_upper_entry.grid(row=6, column=4, padx=5, pady=5, sticky="w")

        # Diagnostic Info
        diag_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        diag_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Label(diag_frame, text="Fitting Details:", font=self.font).pack(side=tk.TOP, anchor="w")
        self.diag_text = tk.Text(diag_frame, height=10, font=self.font)
        self.diag_text.pack(fill=tk.X)
        self.diag_text.insert(tk.END, "No details yet.")
        self.diag_text.config(state=tk.DISABLED)

        # Plotting Area
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data placeholders
        self.filename = None
        self.V_data = None
        self.i_data = None
        self.E_rev = None
        self.fit_results = None
        self.eta_kin = None
        self.Eta_ohm = None
        self.Eta_RCL = None
        self.Eta_res = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        self.y5 = None
        self.y6 = None

    def choose_file(self):
        filetypes = [("Excel files", "*.xlsx"), ("Text files", "*.txt")]
        filename = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".xlsx", ".txt"]:
                messagebox.showerror("File Error", "Only .txt and .xlsx files are supported.")
                return
            self.filename = filename
            self.file_label.config(text=filename)
        else:
            self.filename = None
            self.file_label.config(text="None")

    def perform_fit(self):
        if not self.filename:
            messagebox.showerror("File Error", "No file selected!")
            return
        try:
            T = float(self.temp_entry.get())
            R_CL = float(self.rcl_entry.get())
            HFR = float(self.hfr_entry.get())
            pH_val = float(self.pH_entry.get())
            i_lower = float(self.tafel_lower_entry.get())
            i_upper = float(self.tafel_upper_entry.get())
            if i_lower >= i_upper:
                messagebox.showerror("Input Error", "Tafel-fit range: Lower limit must be < Upper limit.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
            return

        # Compute the thermodynamic potential for water splitting.
        self.E_rev = 1.2291 - 0.0008456 * (T - 298.15)

        # Read raw data and perform reference electrode conversion.
        V_raw, i_raw = read_data(self.filename)
        if V_raw is None or i_raw is None:
            return
        ref_choice = self.ref_var.get()
        offset = self.reference_options[ref_choice]
        if ref_choice == "RHE":
            V_corrected_for_ref = V_raw
        else:
            V_corrected_for_ref = V_raw + 0.0592 * pH_val + offset

        self.V_data = V_corrected_for_ref
        self.i_data = i_raw

        # Iterative Tafel fitting incorporating the accumulation condition.
        iterative_result = iterative_tafel_fit(self.V_data, self.i_data, self.E_rev, HFR, R_CL, i_lower, i_upper)
        if iterative_result is None:
            return
        fit_result, Eta_ohm_all, Eta_RCL_all, accumulation_all = iterative_result
        b_kin = fit_result["b_kin"]
        i0 = fit_result["i0"]
        self.fit_results = fit_result

        # Calculate kinetic overpotential from the fitted parameters.
        self.eta_kin = b_kin * np.log10(self.i_data / i0)
        # Use the iterative results for ohmic and R_CL components.
        self.Eta_ohm = Eta_ohm_all
        self.Eta_RCL = Eta_RCL_all

        # Calculate residual overpotential and enforce non-negativity.
        self.Eta_res = (self.V_data - self.E_rev) - (self.eta_kin + self.Eta_ohm + self.Eta_RCL)
        self.Eta_res = np.maximum(self.Eta_res, 0)

        # Build the six curves for plotting.
        self.y1 = np.full_like(self.i_data, self.E_rev)
        self.y2 = self.y1 + self.eta_kin
        self.y3 = self.y2 + self.Eta_ohm
        self.y4 = self.y3 + self.Eta_RCL
        self.y5 = self.y4 + self.Eta_res
        self.y6 = self.V_data

        # Update diagnostic details.
        details = fit_result["candidate_details"]
        diag_info = (
            f"Kinetic Fit (iterative):\n"
            f"  Tafel slope (b_kin): {b_kin:.4f} V/dec\n"
            f"  Exchange current density (i0): {i0:.4e} A/cm²\n"
            f"  Intercept: {fit_result['intercept']:.4f} V\n"
            f"  R²: {fit_result['r_squared']:.4f}\n"
            f"  Data points in {i_lower:.4g}–{i_upper:.4g} A/cm²: {fit_result['n_points']}\n\n"
            f"Best Candidate 60 mV Window:\n"
            f"  Voltage range: {details['window'][0]:.4f} V to {details['window'][1]:.4f} V\n"
            f"  (Center = {details['center']:.4f} V)\n"
            f"  Subwindows:\n"
            f"     20 mV: {details['subwindows']['20mV'][0]:.4f} – {details['subwindows']['20mV'][1]:.4f} (n={details['counts']['20mV']})\n"
            f"     40 mV: {details['subwindows']['40mV'][0]:.4f} – {details['subwindows']['40mV'][1]:.4f} (n={details['counts']['40mV']})\n"
            f"     60 mV: {details['subwindows']['60mV'][0]:.4f} – {details['subwindows']['60mV'][1]:.4f} (n={details['counts']['60mV']})\n"
            f"  Rel. SE% (b): {details['rel_err_b'] * 100:.2f}%\n"
            f"  Rel. SE% (i0): {details['rel_err_i0'] * 100:.2f}%\n"
            f"  Combined Metric: {details['candidate_metric'] * 100:.2f}%\n"
        )
        self.diag_text.config(state=tk.NORMAL)
        self.diag_text.delete("1.0", tk.END)
        self.diag_text.insert(tk.END, diag_info)
        self.diag_text.config(state=tk.DISABLED)

        # Plotting the curves.
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xscale("log" if self.x_log_scale else "linear")
        tailwind_colors = [
            "#080c0c",  # Night
            "#283d3b",  # Dark slate gray
            "#197278",  # Caribbean Current
            "#edddd4",  # Champagne pink
            "#c44536",  # Persian red
            "#772e25"   # Burnt umber
        ]
        line1, = self.ax.plot(self.i_data, self.y1, color=tailwind_colors[0],
                              linestyle="--", linewidth=2, label="E_rev")
        line2, = self.ax.plot(self.i_data, self.y2, color=tailwind_colors[1],
                              linestyle="-", linewidth=2, label="η_kin")
        line3, = self.ax.plot(self.i_data, self.y3, color=tailwind_colors[2],
                              linestyle="-", linewidth=2, label="η_ohm")
        line4, = self.ax.plot(self.i_data, self.y4, color=tailwind_colors[3],
                              linestyle="-", linewidth=2, label="η_RCL")
        line5, = self.ax.plot(self.i_data, self.y5, color=tailwind_colors[4],
                              linestyle="-", linewidth=2, label="η_res")
        line6, = self.ax.plot(self.i_data, self.y6, color=tailwind_colors[5],
                              marker="o", linestyle="none", markersize=5, label="Original LSV (RHE scale)")
        self.lines = {
            "E_rev": line1,
            "E_rev_kin": line2,
            "E_rev_kin_ohm": line3,
            "E_rev_kin_ohm_rcl": line4,
            "E_rev_kin_ohm_rcl_res": line5,
            "Original_LSV": line6
        }
        self.ax.set_xlabel("Current Density (A/cm²)", fontsize=24)
        self.ax.set_ylabel("Potential (V)", fontsize=24)
        self.ax.set_title("LSV Overpotential Analysis", fontsize=24)
        self.ax.legend(fontsize=24)
        self.canvas.draw()

    def export_data(self):
        if self.i_data is None or self.y1 is None:
            messagebox.showerror("Export Error", "No data available. Please perform a fit first.")
            return

        # Sheet 1: Six Curves Data
        df_curves = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "E_rev": self.y1,
            "η_kin": self.y2,
            "η_ohm": self.y3,
            "η_RCL": self.y4,
            "η_res": self.y5,
            "Original LSV (RHE scale)": self.y6
        })

        # Sheet 2: Overpotential Components
        df_components = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "η_kin (V)": self.eta_kin,
            "η_ohm (V)": self.Eta_ohm,
            "η_RCL (V)": self.Eta_RCL,
            "η_res (V)": self.Eta_res
        })

        # Sheet 3: Fitting Info
        # Gather key fitting information from self.fit_results.
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

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")],
                                                 title="Save Plot Data As")
        if not file_path:
            return
        try:
            with pd.ExcelWriter(file_path) as writer:
                df_curves.to_excel(writer, sheet_name="Six Curves", index=False)
                df_components.to_excel(writer, sheet_name="Overpotential Components", index=False)
                df_fit_info.to_excel(writer, sheet_name="Fitting Info", index=False)
            messagebox.showinfo("Export Successful", f"Data exported successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data:\n{e}")

    def toggle_x_scale(self):
        if self.i_data is None or self.y1 is None:
            messagebox.showerror("Plot Error", "No plot available. Please perform a fit first.")
            return
        self.x_log_scale = not self.x_log_scale
        self.ax.set_xscale("log" if self.x_log_scale else "linear")
        if self.x_log_scale:
            self.lines["E_rev"].set_visible(False)
            self.lines["E_rev_kin"].set_visible(True)
            self.lines["E_rev_kin_ohm"].set_visible(False)
            self.lines["E_rev_kin_ohm_rcl"].set_visible(False)
            self.lines["E_rev_kin_ohm_rcl_res"].set_visible(False)
            self.lines["Original_LSV"].set_visible(True)
        else:
            for line in self.lines.values():
                line.set_visible(True)
        handles, labels = self.ax.get_legend_handles_labels()
        visible_handles = [h for h, l in zip(handles, labels) if h.get_visible()]
        visible_labels = [l for h, l in zip(handles, labels) if h.get_visible()]
        self.ax.legend(visible_handles, visible_labels, fontsize=12)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedLSVAnalysisApp(root)
    root.mainloop()
