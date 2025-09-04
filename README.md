# LSV breakdown
A lightweight Tk/ttk desktop app for dissecting LSV curves into kinetic, ohmic (HFR), catalyst-layer (RCL), and residual overpotentials. It auto-hunts a robust Tafel window, iteratively corrects for ohmic and CL drops, and exports publication-ready tables.
## How to Use This File

### What this app does

- Read .xlsx or .txt file with LSV data.
   - For .xlsx, the first column with Votage in unit V. The second column with Current density in unit A cm-2.
   - For .txt, first two columns = Voltage, Current (whitespace-delimited, no header)
- Converts potentials to RHE (built-in offsets for Ag/AgCl, SCE, HgO; pH-aware)
- Computes E_rev(T) with a linear temperature dependence
- Iteratively fits the Tafel slope while subtracting HFR and RCL contributions
- Decomposes the measured potential into η_kin, η_ohm, η_RCL, η_res and shows a stacked reconstruction vs the original LSV
- Exports three Excel sheets: stacked curves, component overpotentials, and fit metadata 

### Quick start
1. Install requirements: pip install numpy pandas matplotlib scipy
2. Run the script: LSVbreak.py
3. A GUI window will appear. Follow these steps
   - **Load Data**: Click the “Browse” button to select your LSV data file
   - **Set Parameters**: Temperature, RCL and HFR are needed before analysis
   - **Reference Electrode & pH**: Use the dropdown menu to pick the reference electrode (RHE, Ag/AgCl, SCE, HgO), and set the pH. If you choose RHE, no offset is added; otherwise, the code converts your data to the RHE scale automatically (applying the offset + 0.0592 * pH)
   - **Tafel Fit Range**: By default, the code uses 0.02–0.8 A/cm² to find the Tafel slope. You can now edit these fields to use any current-density window (e.g., 0.01–0.5 A/cm²)
   - **Overpotential Analysis**: Click “Fit&Plot” button. The program will process the data and plot overpotential components step by step. Toggle X-Axis Scale switches current between linear/log
   - **Data Export**: Click "Export Data" to save an Excel file with three sheets that include all data used for plotting.

### What gets exported
- Curves (stacked): Current Density, E_rev, E_rev + η_kin, + η_ohm, + η_RCL, + η_res, and Original LSV (RHE) — ready to plot as overlays
- Overpotential Components: η_kin (V), η_ohm (V), η_RCL (V), η_res (V) vs current
- Fitting Info: Tafel slope (V/dec), Exchange current density (A/cm²), Intercept (V), R², N points

## How the fitting works

### Overpotential Calculation
- **E = E_rev + η_kin + η_ohm + η_RCL + η_res**

### Iterative correction loop for Tafel fitting
- Subtracts η_ohm = i·HFR
- Computes η_RCL from the current Tafel slope guess b_kin using a utilization law (below)
- Forms η_corr = (V − E_rev) − η_ohm − η_RCL and performs a Tafel fit in log₁₀(i)–η space
- Updates b_kin, checks convergence (|Δb| < 1e−3 V/dec) and a feasibility guard at the upper-bound current (the reconstructed sum must not exceed the available overpotential)

### Robust Tafel-window hunt
Inside the chosen current range, the code slides a 60 mV window and evaluates three nested subwindows (20/40/60 mV). It computes slopes/intercepts for each subwindow and chooses the candidate with the smallest combined relative error of slope and i₀ (mean±std stability metric). This balances bias–variance and avoids cherry-picking ultra-narrow spans

---
### Notes and references
- The Tafel fitting method is inspired by 10.1021/acs.jpcc.9b06820
- Recommended RCL fitting method: https://github.com/NREL/OSIF
- The detailed explanation and measurement of RCL can be found in Articles: 10.1149/1945-7111/acee25 and 10.1021/acscatal.4c02932

---
### How to cite
Release v1.3.0 is archived on Zenodo:
https://doi.org/10.5281/zenodo.17054694

