# LSV Overpotential Analysis

## How to Use This File

### Prerequisites

- A .xlsx or .txt file with LSV data.
   - The first column with Votage in unit V.
   - The second column with Current density in unit A cm-2.

### Running the Program
1. Open the Python script in an environment that supports Tkinter (e.g., IDLE, Anaconda, or a standard Python environment).
2. Run the script: Vbreak001.py
3. A GUI window will appear. Follow these steps:
   - **Load Data**: Click the “Browse” button to select your LSV data file. The file should have suffix .xslx.
   - **Set Parameters**: Temperature, RCL and HFR are needed before analysis.
   - **Reference Electrode & pH**: Use the dropdown menu to pick the reference electrode (RHE, Ag/AgCl, SCE, HgO), and set the pH. If you choose RHE, no offset is added; otherwise, the code converts your data to the RHE scale automatically (applying the offset + 0.0592 * pH).
   - **Tafel Fit Range**: By default, the code uses 0.005–0.1 A/cm² to find the Tafel slope. You can now edit these fields to use any current-density window (e.g., 0.01–0.5 A/cm²).
   - **Overpotential Analysis**: Click “Fit&Plot” button. The program will process the data and plot overpotential components step by step.
   - **Data Export**: Click "Export Data" to save an Excel file with two sheets that include all data used for plotting.

## Explanation of the Code

### Overpotential Calculation
- **E = E_rev + η_kin + η_ohm + η_RCL + η_res**

### Tafel Slope Analysis
- **Data Selection**: Randomly select multiple 60mV-wide voltage windows as data sets for fitting.
- **Multi-Level Fitting**: Fit the Tafel equation for each data set three times using subsets of 20, 40 and 60mV
- **Statistical Analysis**: Evaluate the standard error percentage (SE%) for b and i₀.
- **Best Fit Selection**: The data set with the smallest SE% is chosen.
- Iteratively fits the Tafel slope while removing the influence of the catalyst-layer (R_CL) overpotential. In the first iteration only the ohmic (HFR) drop is subtracted.

---
### Notes and references
- For detailed explanation and formula, see pdf file "Intro".
- The Tafel fitting method is inspired by 10.1021/acs.jpcc.9b06820. I am not sure if it's accurate or if the author truly means it. However, this "expanding window size" idea does help to find a relative convincing Tafel slope.
- Recommended RCL fitting method: https://github.com/NREL/OSIF.
- The detailed explanation and measurement of RCL can be found in Articles: 10.1149/1945-7111/acee25 and 10.1021/acscatal.4c02932. 

### Contributions
Feel free to fork this repository, make improvements, and submit pull requests!
