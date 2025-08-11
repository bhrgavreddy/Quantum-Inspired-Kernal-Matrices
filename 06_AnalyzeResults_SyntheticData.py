"""
Quantum-Inspired Results Analysis and LaTeX Table Generation for Synthetic Datasets

Author: Dr. techn Sebastian Raubitzek MSc. BSc.

This Python script analyzes results from quantum-inspired kernel matrix experiments, comparing the performance of different symmetry group-based SVM classifiers with the classical CatBoost model. The results are loaded from a JSON file, converted to a pandas DataFrame, and used to dynamically generate LaTeX tables for easy comparison of accuracy, precision, recall, and F1-score metrics across multiple datasets and group types.

### Key Components:
1. **Loading and Normalizing JSON Data**:
   - The script begins by loading the results from a JSON file (`results_log_scale1_2024_synth_fin.json`). This file contains performance metrics for various classifiers across multiple datasets, including symmetry group-based SVMs and the CatBoost classifier.
   - The JSON data is normalized into a pandas DataFrame for further processing.

2. **Performance Metrics Analysis**:
   - The script evaluates the following performance metrics for each dataset:
     - **Accuracy**: The percentage of correct predictions made by the model.
     - **Precision**: The proportion of true positives out of all predicted positives.
     - **Recall**: The proportion of true positives out of all actual positives.
     - **F1-score**: The harmonic mean of precision and recall, providing a balanced measure of both.
   - For each metric, the script compares the performance of symmetry group-based SVMs with CatBoost as a baseline.

3. **Data Processing**:
   - **Pivoting SVM Data**: For each dataset, the SVM results are pivoted to show symmetry group types as columns, allowing for easy comparison between group-based feature maps.
   - **Merging CatBoost Results**: The first result for the CatBoost model is extracted and merged with the pivoted SVM data for each dataset, enabling a side-by-side comparison.

4. **LaTeX Table Generation**:
   - The script dynamically generates LaTeX code to create tables for each metric (accuracy, precision, recall, and F1-score). These tables summarize the performance of each classifier across different datasets.
   - **Dynamic Headers**: The LaTeX table headers are created dynamically based on the unique symmetry group types in the dataset.
   - **LaTeX Table Rows**: Each row in the table corresponds to a dataset, with the performance of each symmetry group-based SVM and CatBoost displayed for comparison.

5. **Example Workflow**:
   1. Load the results from the JSON file.
   2. Convert the results to a pandas DataFrame.
   3. Pivot the SVM results to have symmetry group types as columns.
   4. Merge the CatBoost results with the SVM results.
   5. Dynamically generate LaTeX tables for accuracy, precision, recall, and F1-score.
   6. Print the generated LaTeX code, which can be directly used in research papers or reports.

6. **Context and Purpose**:
   - This script is designed to automate the comparison of quantum-inspired SVM classifiers based on symmetry group feature maps with a classical machine learning baseline (CatBoost). The results are presented in a format suitable for academic publication, allowing researchers to assess the effectiveness of different symmetry-based feature maps in various classification tasks.
   - By generating LaTeX tables, this script simplifies the process of including detailed performance comparisons in research papers, especially for experiments involving multiple datasets and classifiers.

7. **Applications**:
   - The generated LaTeX tables can be used to showcase the results of quantum-inspired classification experiments, demonstrating how different symmetry groups influence model performance.
   - The script facilitates a deeper understanding of how symmetry-based transformations affect the kernel matrix and the downstream performance of machine learning models.

### Notes:
- The symmetry group types used in the SVM classifiers include SO, SL, SU, GL, U, and O, each representing different transformations applied to the input data.
- The CatBoost classifier is used as a baseline to evaluate the performance of classical models compared to quantum-inspired approaches.

References:
- Raubitzek et al., 2024, Physics Letters A 525, "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning"
- CatBoost documentation for classical machine learning comparisons.
"""

import json
import numpy as np
import pandas as pd

# Correcting the path and reloading the JSON data
json_path = "results_log_scale1_2024_synth_fin.json"
print(json_path)
# Load JSON data from the file
with open(json_path, 'r') as file:
    json_data = json.load(file)

# Convert JSON data to a pandas DataFrame
df = pd.json_normalize(json_data)

print(json_path)
print("accuracy")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_accuracy = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_accuracy']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_accuracy = df.pivot_table(index='dataset', columns='group_type', values='svm_accuracy', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_accuracy = pd.merge(df_svm_pivot_accuracy, df_catboost_first_per_dataset_accuracy, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_accuracy = "\n".join(df_final_accuracy.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_accuracy']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_accuracy = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_accuracy}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_accuracy)






print(json_path)
print("precision")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_precision = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_precision']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_precision = df.pivot_table(index='dataset', columns='group_type', values='svm_precision', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_precision = pd.merge(df_svm_pivot_precision, df_catboost_first_per_dataset_precision, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_precision = "\n".join(df_final_precision.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_precision']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_precision = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_precision}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_precision)








print(json_path)
print("recall")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_recall = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_recall']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_recall = df.pivot_table(index='dataset', columns='group_type', values='svm_recall', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_recall = pd.merge(df_svm_pivot_recall, df_catboost_first_per_dataset_recall, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_recall = "\n".join(df_final_recall.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_recall']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_recall = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_recall}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_recall)



print(json_path)
print("f1_score")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_f1_score = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_f1_score']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_f1_score = df.pivot_table(index='dataset', columns='group_type', values='svm_f1_score', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_f1_score = pd.merge(df_svm_pivot_f1_score, df_catboost_first_per_dataset_f1_score, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_f1_score = "\n".join(df_final_f1_score.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_f1_score']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_f1_score = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_f1_score}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_f1_score)