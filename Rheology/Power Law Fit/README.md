# Power Law Fitting and Data

## Overview

This tool is designed to **scan folders for Excel files**, extract relevant data (η, γ, τ), fit a **linearised power law model** to the data, and collect the results in a structured Excel output. The data must come from the Kinexus Rheometer or must display a simalr result structure.

## Code Structure

The code is divided into three main components:

1. **Folder Scanner:** Scans the input folder for Excel files, extracts the required columns, and fits the power law model to each dataset.
2. **Power Law Fitter:** Performs the actual fitting of the power law model and generates plots for each fit.
3. **Result Writer:** Collects and organizes the results, writing them to an Excel file with both individual sample data and grouped statistics.

---

## Folder Structure

- **`Evaluate/Input`:** Place your Excel files here. The script will automatically detect these columns, even if the header row is not the first row.
- **`Evaluate/Output/Data collection.xlsx`:** The final Excel file containing all results, including individual sample fits and grouped statistics.
- **`Evaluate/Plots`:** Automatically generated plots for each fit, showing the power law fit and the data points.

---

## Power Law Model

The power law model used is:

\[
\eta = K \cdot \gamma^{n-1}
\]

Where:
- **η** is the apparent viscosity (Pa·s)
- **γ** is the shear rate (s⁻¹)
- **K** is the consistency index (Pa·sⁿ)
- **n** is the flow behavior index (dimensionless)

### Linearisation

To fit the model, the equation is linearised by taking the natural logarithm of both sides:

\[
\ln(\eta) = \ln(K) + (n-1) \cdot \ln(\gamma)
\]

This allows the use of linear regression to estimate the parameters **K** and **n**.

---

## Data Processing

### Input Data
- The script expects Excel files with columns for `eta`, `gamma`, and `tau`.
- It automatically detects the correct header row and column names, even if they are not in the first row or are slightly misspelled.
- Only data points with valid, positive values for `eta` and `gamma` are used for fitting. Points with `gamma > 40` are excluded to avoid fitting the high-shear tail.

### Output Data
- For each sample, the script calculates and saves:
  - **K** (consistency index)
  - **n** (flow behavior index)
  - **R²** (coefficient of determination, indicating the goodness of fit)
- Results are grouped by sample and by experimental groups (e.g., DOE numbers), with mean and standard deviation calculated for each group.

---

## Usage

1. Place your Excel files in the `Evaluate/Input` folder.
2. Run the script. It will automatically:
   - Scan the input folder
   - Fit the power law model to each dataset
   - Generate plots for each fit
   - Write the results to `Evaluate/Output/Data collection.xlsx`
3. Check the output Excel file and the `Evaluate/Plots` folder for results and visualizations.

---

## Notes

- The script uses **η (apparent viscosity)** for the fit, not σ (shear stress).
- The script is robust to missing or malformed data and will skip files or rows that cannot be processed.
- The output Excel file includes both individual sample results and grouped statistics, making it easy to compare results across different experimental conditions.
