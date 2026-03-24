# Concentration Prediction Pipelines (03.2026)

## Overview

This repository contains several approaches of pipelines for predicting concentration from morphological and rheological features. Mainly the pipelines using a Gaussian Process Regression (GPR), but also a linear model for comparison (a linear multivariate ridge regression model). There are multiple versions of the GPR pipeline, as they differ in using both the morphological and rheological data or each individually. Adittionaly there is a pipeline for a sensitivity analysis using a derivation of the Fisher Information Matrix (FIM). The system is currently optimised on the data used by OHi, thus, changes in the input file structure might also require changes in the script.

## Overview Modelling file

1. **Data** (Morphological data)
2. **GPR** (All scripts for the GPR (Gaussian Process Regression) pipeline)
3. **GPR - Morph only** (All scripts for the GPR pipeline, but only the morphological data is used for modelling)
4. **GPR - Rheo only** (All scripts for the GPR pipeline, but only the rheological data is used for modelling)
5. **Linear Model** (Simalar scripts to GPR, only the modelling type is exchanged for a linear system)
6. **Sensibility Analysis** (Basic script set for the FIM based sensibility analysis)
7. **Master Sheet - Change** (Excel file in which rheological and pigment data, as well as some additional information about the data can be found)

## Example structure of GPR
The subsequent section of the README gives an overview on the GPR structure, which can then be translated to the other files. The script itself includes many additional informations as well. If any further comprehension is demanded, pls contact the author of the scripts.



### Pipeline Architecture

The pipeline consists of several key components:

1. **Data Loading & Cleaning** (`MatrixGatherer.py`)
2. **Data Augmentation** (`synthetic_data.py`, `DataGeneration.py`)
3. **Feature Processing** (`Processor.py`)
4. **Model Training & Optimization** (main script)
5. **Evaluation & Results Export**

### Key Files

#### 1. Main Pipeline Script

``` python
"""
End-to-end pipeline for learning concentration from morphology and rheology using Gaussian Process Regression.
"""
```
Purpose: Orchestrates the entire workflow from data loading to model evaluation.
Key Features:
- Loads and aligns morphology, rheology, and concentration data
- Performs data augmentation using measurement uncertainty
- Applies feature preprocessing and weighting
- Uses Bayesian Optimization and grid search for hyperparameter tuning
- Trains and evaluates the final GP model
- Exports predictions, landscapes, and evaluation metrics

Output Files:
- predictions_diameter_fixed.csv: Test set predictions with uncertainties
- parameter_ablation.csv: Results of parameter ablation study
- combined_weight_landscape_diameter_fixed.csv: Grid search results
- grid_stage1_coarse.csv: Coarse grid search results


#### 2. Data Augmentation (DataGeneration.py)

Generates synthetic data from measurement uncertainty to augment the training set.


Key Methods:
- generate(): Creates synthetic morphology data
- generate_rheo(): Creates synthetic rheology data


Augmentation Logic:
- Determines number of synthetic samples based on target uncertainty
- Uses normal distribution to generate synthetic values
- Preserves volume fractions while varying other features
- Appends "_syn{N}" to sample names for synthetic data


Uncertainty-Based Generation:

```python
def determine_synthetic_count(mean, sd):
    r = sd / mean
    if 1/(r) <= 1:
        return 0
    elif 1/(r**2) < 20:
        return round(1/(r**2))
    else:
        return 20

```





#### 3. Synthetic Data Generator (synthetic_data.py)

Generates synthetic morphology vectors and concentration values.


Key Features:
- Uses normal distribution with mean and standard deviation
- Handles zero/negative standard deviations by using 30% of mean
- Clips negative values to zero for physical meaningfulness
- Maintains volume fractions while generating synthetic morphology

Core Methods:
- expand_morphology(): Generates synthetic morphology vectors
- expand_concentration(): Generates synthetic concentration values


#### 4. Feature Processor (Processor.py)

Preprocesses features through normalization and weighting.

Key Functionality:
- Column-wise Normalization: Scales features by their maximum values
- Volume Fraction Weighting: Weights features by their volume fraction
- Manual Weight Application: Applies user-specified weights to features

Processing Pipeline:
- Fit: Computes maximum values for normalization
- Transform: Applies normalization
- Weighting: Applies volume fraction or manual weights

Special Handling:
- Avoids division by zero in normalization
- Handles both morphology and rheology features separately



### Pipeline Stages

1. Data Loading and Cleaning
- Loads morphology, rheology, and concentration data
- Aligns samples across all data sources
- Filters out invalid samples

2. Data Augmentation
- Generates synthetic samples based on measurement uncertainty
- Number of synthetic samples depends on relative uncertainty
- Preserves original sample distribution characteristics

3. Feature Preprocessing
- Normalizes features column-wise
- Applies volume fraction weighting
- Prepares features for model training

4. Parameter Ablation Study
- Tests individual feature importance (1D models)
- Performs leave-one-out analysis using Bayesian Optimization
- Identifies most informative features

5. Hyperparameter Optimization
- Uses Bayesian Optimization to find optimal feature weights
- Builds structured performance landscape with grid search
- Combines BO optimum with historically strong regions

6. Final Model Training
- Trains GP model with optimal weights
- Evaluates on held-out test set
- Computes RMSE and R² metrics
- Performs cross-validation for robustness assessment

7. Results Export
- Saves predictions with uncertainties
- Exports optimization landscapes
- Stores ablation study results


### Usage Instructions

#### Requirements
- Python 3.8+
- Required packages: numpy, pandas, scikit-learn, scikit-optimize, tqdm
- Input data in specified folder structure

Running the Pipeline
- Place morphology data in Data/ folder
- Ensure Excel file with rheology/concentration is available
- Configure paths in the main script
- Run the main script:


#### Data Structure Requirements

Input Files

Morphology Data:
- Located in Data/ folder
- Subfolders for each sample
- CSV files named PSD_cluster_summary_final_k{cluster_count}_Volume.csv
- Required columns: Mean_Diameter, Std_Diameter, Mean_Circularity, Std_Circularity, Mean_Compactness, - - Std_Compactness, Total_VolumeFraction

Rheology/Concentration Data:
- Excel file with specific column structure
- Sample names should start with "DOE "
- Required columns for rheology: mean1 (col 20), sd1 (col 22), mean2 (col 21), sd2 (col 23)
- Required columns for concentration: mean (col 3), sd (col 4)

Output Files

All output files are saved in the script's directory:
- Predictions: predictions_diameter_fixed.csv
- Ablation results: parameter_ablation.csv
- Optimization landscape: combined_weight_landscape_diameter_fixed.csv
- Grid search results: grid_stage1_coarse.csv
