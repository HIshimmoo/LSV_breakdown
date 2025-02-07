import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import linregress
import random
import os

# ---------------------------
# Helper functions for Tafel analysis (new method with SE% consideration)
# ---------------------------
def read_data(filename):
    """
    Reads the first two columns from either a .txt or .xlsx file.
    Returns (V, i) as NumPy arrays. Returns (None, None) if there's an error.
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".xlsx":
            data = pd.read_excel(filename)
        elif ext == ".txt":
            # Assuming whitespace-delimited file with no header, and at least 2 columns
            data = pd.read_csv(filename, delim_whitespace=True, header=None)
        else:
            messagebox.showerror("File Error", "Only .txt and .xlsx files are supported.")
            return None, None
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading the file:\n{e}")
        return None, None

    # Check that we have at least 2 columns
    if data.shape[1] < 2:
        messagebox.showerror("Column Error", "The file must have at least two columns (Voltage, Current).")
        return None, None

    # Extract columns
    V = data.iloc[:, 0]
    i = data.iloc[:, 1]

    # Basic cleaning: drop invalid or missing values
    mask = (i > 0) & V.notnull() & i.notnull()
    V = V[mask].to_numpy()
    i = i[mask].to_numpy()

    if len(V) == 0:
        messagebox.showerror("Data Error", "No valid data found (check for positive currents, no NaN).")
        return None, None

    return V, i


def fit_tafel(eta_corr, i, i_lower=0.005, i_upper=0.1):
    """
    Performs the Tafel fitting routine with subwindow checks,
    restricted to the user-chosen current density range i_lower to i_upper.
    """
    # Restrict to user-chosen current density range
    mask = (i >= i_lower) & (i <= i_upper)
    if np.sum(mask) < 5:
        messagebox.showerror(
            "Data Error",
            f"Not enough data points in the chosen current-density range: "
            f"{i_lower:.4g}–{i_upper:.4g} A/cm²."
        )
        return None

    eta_sel = eta_corr[mask]
    i_sel = i[mask]

    # Determine overall voltage (eta_corr) range.
    vmin, vmax = np.min(eta_sel), np.max(eta_sel)
    window_width = 0.06  # 60 mV
    if vmax - vmin < window_width:
        messagebox.showerror("Data Error",
                             "The voltage range in the selected current region is too narrow for a 60 mV window.")
        return None

    candidate_results = []
    # Randomly choose 1000 candidate 60 mV windows (the code comment says 10, but let's keep 1000).
    for candidate in range(1000):
        start = random.uniform(vmin, vmax - window_width)
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
            n = np.sum(sub_mask)
            counts[key] = n
            if n < 3:
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

        # Compute i0 for each subwindow.
        i0s = np.array([10 ** (-intercepts[j] / slopes[j]) for j in range(len(slopes))])
        std_slope = np.std(slopes)
        avg_slope = np.mean(slopes)
        std_i0 = np.std(i0s)
        avg_i0 = np.mean(i0s)
        # Relative errors (SE%):
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
            "counts": counts
        })

    if not candidate_results:
        messagebox.showerror("Fitting Error", "Could not obtain valid fits in any candidate 60 mV window.")
        return None

    # Choose candidate with the smallest candidate_metric (i.e. smallest average relative error).
    best_candidate = min(candidate_results, key=lambda x: x["candidate_metric"])

    result = {
        "b_kin": best_candidate["avg_slope"],
        "intercept": best_candidate["avg_intercept"],
        "i0": 10 ** (-best_candidate["avg_intercept"] / best_candidate["avg_slope"]),
        "r_squared": best_candidate["avg_r2"],
        "n_points": np.sum((i >= 0.005) & (i <= 0.1)),
        "candidate_details": best_candidate
    }
    return result


def calculate_i0(b_kin, intercept):
    """
    Calculates the exchange current density i0 from Tafel slope (b_kin) and intercept.
    """
    return 10 ** (-intercept / b_kin)


# ---------------------------
# Integrated GUI Application
# ---------------------------
class IntegratedLSVAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Integrated LSV Overpotential Analysis")
        master.geometry("1400x900")

        # Define a common font for the entire window
        self.font = ("Arial", 24)

        # Reference electrode offsets
        self.reference_options = {
            "RHE": 0.0,
            "Ag/AgCl (sat)": 0.197,
            "SCE (sat)": 0.242,
            "HgO (sat)": 0.098
        }

        # ---------------------------
        # Top Frame: Input Parameters
        # ---------------------------
        input_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 1) File browsing
        tk.Label(input_frame, text="Selected File:", font=self.font).grid(row=0, column=0, sticky="w")
        self.file_label = tk.Label(input_frame, text="None", font=self.font)
        self.file_label.grid(row=0, column=1, sticky="w", padx=5)
        self.browse_button = tk.Button(input_frame, text="Browse", font=self.font,
                                       command=self.choose_file, bg="#e6f2ff")
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Instruction about file types
        tk.Label(input_frame, text="(Allowed: .txt or .xlsx)", font=self.font, fg="blue").grid(row=0, column=3, sticky="w")

        # 2) A single row for Reference Electrode and pH
        tk.Label(input_frame, text="Reference Electrode:", font=self.font).grid(row=1, column=0, sticky="w", pady=5)
        self.ref_var = tk.StringVar(value="RHE")
        self.ref_menu = tk.OptionMenu(input_frame, self.ref_var, *self.reference_options.keys())
        self.ref_menu.config(font=self.font, width=15, bg="#ffffcc")
        self.ref_menu.grid(row=1, column=1, padx=5, sticky="w")
        # ALSO change the dropdown menu (the items) font:
        menu = self.ref_menu["menu"]
        menu.config(font=("Arial", 24))

        tk.Label(input_frame, text="pH:", font=self.font).grid(row=1, column=2, sticky="w", pady=5)
        self.pH_entry = tk.Entry(input_frame, font=self.font, width=6)
        self.pH_entry.insert(0, "0")  # default pH=0
        self.pH_entry.grid(row=1, column=3, padx=5, sticky="w")

        # 3) Temperature
        tk.Label(input_frame, text="Temperature (K):", font=self.font).grid(row=2, column=0, sticky="w")
        self.temp_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.temp_entry.insert(0, "298.15")
        self.temp_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 4) R_CL
        tk.Label(input_frame, text="R_CL (Ohm·cm²):", font=self.font).grid(row=3, column=0, sticky="w")
        self.rcl_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.rcl_entry.insert(0, "0.1")
        self.rcl_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # 5) HFR
        tk.Label(input_frame, text="HFR (Ohm·cm²):", font=self.font).grid(row=3, column=2, sticky="w")
        self.hfr_entry = tk.Entry(input_frame, font=self.font, width=10)
        self.hfr_entry.insert(0, "0.01")
        self.hfr_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # 6) Fit & Plot button
        self.fit_button = tk.Button(input_frame, text="Fit & Plot", font=self.font,
                                    command=self.perform_fit, bg="#cce6ff")
        self.fit_button.grid(row=4, column=0, columnspan=2, pady=10)

        # 7) Export button
        self.export_button = tk.Button(input_frame, text="Export Data", font=self.font,
                                       command=self.export_data, bg="#d9ead3")
        self.export_button.grid(row=4, column=2, columnspan=2, pady=10)

        # 8) Toggle X-axis scale button
        self.x_log_scale = False  # Track whether we're in log scale
        self.toggle_xscale_button = tk.Button(input_frame, text="Toggle X-Axis Scale", font=self.font,
                                              command=self.toggle_x_scale, bg="#ffe699")
        self.toggle_xscale_button.grid(row=5, column=0, columnspan=4, pady=5)

        # Add Tafel Fit Range inputs (row 6 for instance)
        tk.Label(input_frame, text="Tafel Fit Range:", font=self.font).grid(row=6, column=0, sticky="w", pady=5)

        # Lower limit entry
        tk.Label(input_frame, text="Lower (A/cm²):", font=self.font).grid(row=6, column=1, sticky="e")
        self.tafel_lower_entry = tk.Entry(input_frame, font=self.font, width=8)
        self.tafel_lower_entry.insert(0, "0.005")  # default
        self.tafel_lower_entry.grid(row=6, column=2, padx=5, pady=5, sticky="w")

        # Upper limit entry
        tk.Label(input_frame, text="Upper (A/cm²):", font=self.font).grid(row=6, column=3, sticky="e")
        self.tafel_upper_entry = tk.Entry(input_frame, font=self.font, width=8)
        self.tafel_upper_entry.insert(0, "0.1")  # default
        self.tafel_upper_entry.grid(row=6, column=4, padx=5, pady=5, sticky="w")

        # ---------------------------
        # Diagnostic Info
        # ---------------------------
        diag_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        diag_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Label(diag_frame, text="Fitting Details:", font=self.font).pack(side=tk.TOP, anchor="w")
        self.diag_text = tk.Text(diag_frame, height=10, font=self.font)
        self.diag_text.pack(fill=tk.X)
        self.diag_text.insert(tk.END, "No details yet.")
        self.diag_text.config(state=tk.DISABLED)

        # ---------------------------
        # Plotting Area
        # ---------------------------
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
        filetypes = [
            ("Excel files", "*.xlsx"),
            ("Text files", "*.txt")
        ]
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=filetypes
        )
        if filename:
            # Simple check of extension
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
            # New: user-chosen Tafel range
            i_lower = float(self.tafel_lower_entry.get())
            i_upper = float(self.tafel_upper_entry.get())
            # Basic check
            if i_lower >= i_upper:
                messagebox.showerror("Input Error", "Tafel-fit range: Lower limit must be < Upper limit.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
            return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters (T, R_CL, HFR, pH).")
            return

        # E_rev (thermodynamic potential) for water splitting (at T)
        # Slightly simplified: 1.2291 - 0.0008456 * (T - 298.15)
        self.E_rev = 1.2291 - 0.0008456 * (T - 298.15)

        # Read raw data
        V_raw, i_raw = read_data(self.filename)
        if V_raw is None or i_raw is None:
            return

        # Convert to RHE scale if needed
        ref_choice = self.ref_var.get()
        offset = self.reference_options[ref_choice]
        if ref_choice == "RHE":
            # Already on RHE scale
            V_corrected_for_ref = V_raw
        else:
            # E_RHE = E_raw + 0.0592 * pH + offset
            V_corrected_for_ref = V_raw + 0.0592 * pH_val + offset

        # Store final V_data and i_data
        self.V_data = V_corrected_for_ref
        self.i_data = i_raw

        # Next: ohmic correction (HFR) for Tafel fitting
        self.Eta_ohm = self.i_data * HFR
        V_corr = self.V_data - self.Eta_ohm
        eta_corr = V_corr - self.E_rev

        # Tafel fit
        kinetic_fit = fit_tafel(eta_corr, self.i_data, i_lower, i_upper)
        if kinetic_fit is None:
            return
        b_kin = kinetic_fit["b_kin"]
        intercept = kinetic_fit["intercept"]
        i0 = kinetic_fit["i0"]

        # Kinetic overpotential
        self.eta_kin = b_kin * np.log10(self.i_data / i0)

        # Catalyst-layer model for R_CL
        # (Same approach from your original code)
        term = (self.i_data * np.log(10) * R_CL) / (2 * b_kin)
        term = np.clip(term, 0, None)
        U_J = (1 + term ** (1.1982)) ** (-1 / 1.1982)
        self.Eta_RCL = -b_kin * np.log10(U_J)

        # "Residual" overpotential
        self.Eta_res = (self.V_data - self.E_rev) - (self.eta_kin + self.Eta_ohm + self.Eta_RCL)

        # Construct the 6 curves
        self.y1 = np.full_like(self.i_data, self.E_rev)
        self.y2 = self.y1 + self.eta_kin
        self.y3 = self.y2 + self.Eta_ohm
        self.y4 = self.y3 + self.Eta_RCL
        self.y5 = self.y4 + self.Eta_res
        self.y6 = self.V_data  # Original (after ref conversion, if any)

        # Show the details in the text box
        details = kinetic_fit["candidate_details"]
        diag_info = (
            f"Kinetic Fit (ohmic-corrected):\n"
            f"  Tafel slope (b_kin): {b_kin:.4f} V/dec\n"
            f"  Exchange current density (i0): {i0:.4e} A/cm²\n"
            f"  R²: {kinetic_fit['r_squared']:.4f}\n"
            f"  Data points in 0.005–0.1 A/cm²: {kinetic_fit['n_points']}\n\n"
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

        # Plot
        self.ax.clear()
        self.ax.grid(True)

        # Respect current self.x_log_scale for plotting
        self.ax.set_xscale("log" if self.x_log_scale else "linear")

        # 1. Define your palette (as an ordered list of hex codes)
        tailwind_colors = [
            "#080c0c",  # Night
            "#283d3b",  # Dark slate gray
            "#197278",  # Caribbean Current
            "#edddd4",  # Champagne pink
            "#c44536",  # Persian red
            "#772e25"  # Burnt umber
        ]

        # 2. When you plot each curve in perform_fit, set the color:
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

        # Save these line objects in a dict so we can hide/show them later
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

        df_curves = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "E_rev": self.y1,
            "E_rev + η_kin": self.y2,
            "E_rev + η_kin + η_ohm": self.y3,
            "E_rev + η_kin + η_ohm + η_RCL": self.y4,
            "E_rev + η_kin + η_ohm + η_RCL + η_res": self.y5,
            "Original LSV (RHE scale)": self.y6
        })

        df_components = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "η_kin (V)": self.eta_kin,
            "η_ohm (V)": self.Eta_ohm,
            "η_RCL (V)": self.Eta_RCL,
            "η_res (V)": self.Eta_res
        })

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Plot Data As"
        )
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path) as writer:
                df_curves.to_excel(writer, sheet_name="Six Curves", index=False)
                df_components.to_excel(writer, sheet_name="Overpotential Components", index=False)
            messagebox.showinfo("Export Successful", f"Data exported successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data:\n{e}")

    def toggle_x_scale(self):
        """
        Toggles the x-axis between linear and log scale.
        In log scale, only show E_rev + η_kin and the Original LSV.
        In linear scale, show all curves.
        """
        if self.i_data is None or self.y1 is None:
            messagebox.showerror("Plot Error", "No plot available. Please perform a fit first.")
            return

        # Flip the state
        self.x_log_scale = not self.x_log_scale
        self.ax.set_xscale('log' if self.x_log_scale else 'linear')

        if self.x_log_scale:
            # Only show "E_rev + η_kin" and "Original LSV"
            self.lines["E_rev"].set_visible(False)
            self.lines["E_rev_kin"].set_visible(True)
            self.lines["E_rev_kin_ohm"].set_visible(False)
            self.lines["E_rev_kin_ohm_rcl"].set_visible(False)
            self.lines["E_rev_kin_ohm_rcl_res"].set_visible(False)
            self.lines["Original_LSV"].set_visible(True)
        else:
            # Show all lines in linear scale
            for line in self.lines.values():
                line.set_visible(True)

        # get only visible lines/labels
        handles, labels = self.ax.get_legend_handles_labels()
        visible_handles = []
        visible_labels = []
        for h, l in zip(handles, labels):
            if h.get_visible():
                visible_handles.append(h)
                visible_labels.append(l)

        self.ax.legend(visible_handles, visible_labels, fontsize=12)

        # Recompute limits & redraw
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedLSVAnalysisApp(root)
    root.mainloop()
