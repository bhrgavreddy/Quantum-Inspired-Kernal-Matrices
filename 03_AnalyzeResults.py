"""
Quantum-Inspired Results Analysis and LaTeX Table Generation for Classification Metrics

Author: Dr. techn Sebastian Raubitzek MSc. BSc.

This Python script processes the results from quantum-inspired kernel matrix experiments, focusing on the comparison between Support Vector Machine (SVM) classifiers based on different symmetry group feature maps and the classical CatBoost classifier. It reads JSON-formatted results, normalizes them into a pandas DataFrame, and generates tables summarizing the performance across various datasets for different evaluation metrics (accuracy, precision, recall, F1-score).

### Key Components:
1. **Loading and Normalizing JSON Data**:
   - The script loads the result logs stored in JSON format, which contain performance data (accuracy, precision, recall, F1-score) for various symmetry group-based SVMs and the CatBoost classifier across multiple datasets. The JSON data is normalized and converted into a pandas DataFrame for further manipulation.

2. **Performance Evaluation**:
   - The script computes and compares performance metrics for SVM classifiers utilizing feature maps derived from various symmetry groups (SO, SL, SU, GL, U, O) with a CatBoost baseline for each dataset.
   - The metrics include:
     - **Accuracy**: Percentage of correct predictions.
     - **Precision**: Ratio of true positives to predicted positives.
     - **Recall**: Ratio of true positives to actual positives.
     - **F1-Score**: Harmonic mean of precision and recall, offering a balance between the two.

3. **Latex Table Generation**:
   - For each metric (accuracy, precision, recall, and F1-score), the script dynamically generates a LaTeX table.
   - **Pivoting and Merging**: The SVM data is pivoted by symmetry group types (columns), with the CatBoost results appended in the final column.
   - **Dynamic Table Headers**: The script dynamically constructs LaTeX headers based on the symmetry groups included in the dataset.
   - **Latex Table Code**: Complete LaTeX code for each metric’s table is generated and printed.

4. **Metrics Extraction**:
   - For each dataset, the script extracts the first occurrence of the CatBoost result (as a reference) and compares it with the SVM results for each symmetry group.
   - The pivoted and merged data allows for easy comparison between symmetry group-based feature maps and the classical CatBoost approach.

5. **Context and Purpose**:
   - This script builds upon the experimental framework from the paper "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning" (Raubitzek et al., 2024), where quantum kernel estimators based on various Lie groups are compared against classical machine learning approaches.
   - The analysis focuses on demonstrating how symmetry-based feature maps perform in different classification tasks compared to CatBoost, a state-of-the-art classifier, to evaluate the potential quantum advantage.
   - The results are presented in LaTeX tables for publication and further analysis.

### Example Workflow:
1. Load results from a JSON file (`results_log_scale1_2024.json`) containing performance data for SVMs with different symmetry group-based feature maps and CatBoost.
2. Normalize the data into a pandas DataFrame.
3. Pivot the SVM results by symmetry group type and merge them with the CatBoost results.
4. Dynamically generate LaTeX table code for each performance metric (accuracy, precision, recall, F1-score).
5. Print the LaTeX table code for integration into research papers or reports.

### References:
- Raubitzek et al., 2024, Physics Letters A 525, "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning"
- CatBoost documentation: A gradient boosting framework that handles categorical features automatically and provides strong baseline performance.

### Notes:
- The symmetry groups explored in this analysis include SO(n), SL(n), SU(n), GL(n), U(n), and O(n), each offering unique transformations for feature maps in quantum-inspired machine learning.
- The generated LaTeX tables are suitable for inclusion in academic papers, enabling clear comparison of the quantum-inspired approaches with classical models.
"""



import json
import numpy as np
import pandas as pd

# Correcting the path and reloading the JSON data
json_path = "results_log_scale1_2024.json"
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