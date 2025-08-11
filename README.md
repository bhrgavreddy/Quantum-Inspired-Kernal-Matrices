# Quantum_Inspired_Kernel_Matrices

# Lie Group-Based Kernel Methods for Machine Learning with Synthetic Data

This repository contains Python scripts and tools designed to apply Lie group-based kernel methods in machine learning experiments. The project focuses on implementing and verifying quantum-inspired kernel estimators and symmetry-based feature maps for synthetic datasets.

## Authors: Dr. techn. Sebastian Raubitzek MSc. BSc.
# Full Article: https://www.sciencedirect.com/science/article/pii/S0375960124005899

## Overview

This project explores the application of Lie group and symmetry-based transformations to classical datasets for quantum kernel estimations. Various group-theoretic techniques are applied to synthetic data, followed by model training and performance evaluation under different transformations. The key components of this repository include:

1. **Dataset Generation and Preparation**: Scripts for creating and verifying synthetic datasets, ready for machine learning experiments.
2. **Symmetry Feature Map Implementation**: Classes and functions for applying group-based transformations using Lie algebras (e.g., SU(n), SO(n), GL(n)).
3. **Model Experimentation**: Scripts to train machine learning models with transformed datasets and evaluate them under different noise levels and multiplicative factors.
4. **Result Analysis and Verification**: Scripts for analyzing the experimental results and verifying the mathematical properties of applied group-based transformations.

## Repository Structure

├── README.md

├── 01_check_symmetry_properties.py

├── 02_LieGroupQKE_RunExperiments.py

├── 03_AnalyzeResults.py

├── 04_create_synthetic_data.py

├── 05_LieGroupQKE_RunExperimentsSyntheticData.py

├── 06_AnalyzeResults_SyntheticData.py

├── class_quiskit_feature_maps_submission.py

├── class_symmetry_feature_maps_submission.py

├── func_sun.py

├── func_verification.py


All scripts and their corresponding results are part of the research leading to the associated preprint.

## Script Descriptions

1. **01_check_symmetry_properties.py**:
   - **Purpose**: Verifies the symmetry properties of generated Lie group-based feature maps. Ensures that the transformations satisfy the desired mathematical properties such as orthogonality, unitarity, and closure.
   - **Output**: A summary of verification results, providing details about the integrity of the applied transformations.

2. **02_LieGroupQKE_RunExperiments.py**:
   - **Purpose**: Main loop for running machine learning experiments on real datasets using Lie group-based quantum kernel estimators (QKE). Applies group transformations and evaluates model performance.
   - **Output**: A series of results files detailing model performance metrics stored in the `results_log_scale/` directory.

3. **03_AnalyzeResults.py**:
   - **Purpose**: Analyzes the results from machine learning experiments. Generates statistical summaries and visualizations to evaluate the impact of symmetry transformations on model performance.
   - **Output**: Visualizations and LaTeX tables of results stored in the `analysis_results/` directory.

4. **04_create_synthetic_data.py**:
   - **Purpose**: Generates synthetic datasets for experiments. These datasets are specifically designed to test the performance of models under Lie group-based transformations.
   - **Output**: Synthetic datasets saved in the `data/` directory.

5. **05_LieGroupQKE_RunExperimentsSyntheticData.py**:
   - **Purpose**: Applies the same experimental loop as in the real dataset script but focuses on synthetic data. Uses Lie group-based feature maps and quantum kernel estimators to assess model performance.
   - **Output**: Results of synthetic data experiments stored in the `results_synthetic/` directory.

6. **06_AnalyzeResults_SyntheticData.py**:
   - **Purpose**: Analyzes the results from the synthetic dataset experiments. Generates heatmaps and statistical evaluations to summarize the impact of the transformations.
   - **Output**: Visualizations and LaTeX tables stored in the `analysis_results_synthetic/` directory.

7. **class_quiskit_feature_maps_submission.py**:
   - **Purpose**: Contains classes for generating classical feature maps that mimic quantum circuits using Qiskit-inspired transformations.
   - **Output**: Quantum-inspired feature maps applied to the datasets.

8. **class_symmetry_feature_maps_submission.py**:
   - **Purpose**: Defines the `SymmetryFeatureMaps` class, which implements symmetry-based feature transformations using Lie group theory. This class supports transformations like SU(n), SO(n), GL(n), and includes methods for adding noise and modifying datasets.
   - **Output**: Datasets transformed with symmetry-based feature maps.

9. **func_sun.py**:
   - **Purpose**: Provides utility functions for generating SU(n) and related group-based transformations. These functions are used throughout the project for feature map generation.
   - **Output**: Group elements used for dataset transformations.

10. **func_verification.py**:
   - **Purpose**: Contains functions to verify the correctness and mathematical properties (e.g., algebra closure, orthogonality, unitarity) of group transformations.
   - **Output**: Logs detailing the verification results.

## Results

Results from the experiments, including transformed datasets and performance evaluations, are stored in the `results_log_scale/`, `results_synthetic/`, `analysis_results/`, and `analysis_results_synthetic/` directories.

## Prerequisites

Ensure you have the following dependencies installed:

Python 3.6
scikit-learn==0.24.2
catboost==0.26.1
matplotlib==3.2.2
seaborn==0.11.0
pandas==1.1.3
scipy==1.5.3
umap==0.1.1
numpy==1.19.5

## Usage

1. **Symmetry Properties Check**: Run `01_check_symmetry_properties.py` to verify that the symmetry transformations have been correctly applied.
2. **Run Experiments on Real Data**: Use `02_LieGroupQKE_RunExperiments.py` to apply transformations to real datasets and train machine learning models.
3. **Run Experiments on Synthetic Data**: Execute `05_LieGroupQKE_RunExperimentsSyntheticData.py` to run experiments using synthetic datasets.
4. **Analyze Results**: Use `03_AnalyzeResults.py` or `06_AnalyzeResults_SyntheticData.py` to generate summaries, heatmaps, and LaTeX tables.

## License

This project is licensed under the terms of the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).
