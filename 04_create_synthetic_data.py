"""
Synthetic Dataset Generation for Quantum-Inspired Classification Tasks

Author: Dr. techn Sebastian Raubitzek MSc. BSc.

This Python script generates a synthetic dataset specifically designed for quantum-inspired classification tasks, incorporating concepts of feature space mapping through symmetry group transformations. The dataset simulates a binary classification problem where features are generated according to the parameters discussed in the research paper "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning" (Raubitzek et al., 2024).

### Key Components:
1. **Feature Generation**:
    - The dataset consists of three features (`x0`, `x1`, `x2`):
        - `x0`: The first feature is generated to distinguish between two classes using intervals [0, π/3] for class 0 and [2π/3, π] for class 1. This separation aligns with the division of the feature space based on symmetry group transformations.
        - `x1`: This feature is a constant, set to zero for all samples, as used in the experiments involving symmetry groups like SU(2), where only certain features contribute to class separation.
        - `x2`: A uniformly distributed random variable within the range [0, π], which introduces variance into the dataset and allows testing of feature maps that operate on continuous features.

2. **Class Labels**:
    - The dataset represents a binary classification problem with two classes:
        - **Class 0**: Half of the samples, where the `x0` feature is in the interval [0, π/3].
        - **Class 1**: The remaining half of the samples, where the `x0` feature is in the interval [2π/3, π].
    - The class labels are stored in the `Class` column of the dataset.

3. **Shuffling and Data Integrity**:
    - To ensure that the dataset is not ordered, the script shuffles all samples, ensuring randomness and preventing any biases in the dataset's ordering.

4. **Output**:
    - The generated dataset is stored in a pandas DataFrame with columns: `x0`, `x1`, `x2`, and `Class`.
    - The DataFrame is then saved as a CSV file named `synthetic_data_1000.csv` (or whatever number of samples is set by the `n_samples` variable).

5. **Context and Purpose**:
    - This synthetic dataset is designed to test quantum-inspired machine learning models, specifically those utilizing symmetry-based feature maps (such as SO, SL, SU, etc.) described in the paper by Raubitzek et al. The separation of the feature space (`x0`) into distinct intervals corresponds to the encoding of data using Lie group transformations, a central theme in quantum kernel estimation.
    - By generating features within predefined intervals, this dataset allows researchers to test how quantum-inspired feature maps handle feature separation, compression, and class boundary determination in both simple and complex cases.

### Example Workflow:
1. Set the number of samples to generate (`n_samples = 1000`).
2. Generate features `x0`, `x1`, and `x2` as described.
3. Create class labels corresponding to the intervals of `x0`.
4. Shuffle the dataset to remove any ordering.
5. Save the resulting DataFrame as a CSV file for use in classification tasks.

### Applications:
- This dataset is particularly useful for testing quantum-inspired kernel matrices, which encode classical data into higher-dimensional feature spaces using Lie groups.
- It can also be used to benchmark classical models (like CatBoost or SVM) against quantum kernel estimators to evaluate performance differences in symmetry-based data encoding.

### Notes:
- The choice of intervals for `x0` (e.g., [0, π/3] and [2π/3, π]) reflects the approach used in the referenced paper, where specific regions of the feature space are mapped to different classes, representing distinct parts of the Lie manifold.
- Adjustments to the number of features or intervals can be made based on the desired complexity of the classification task.

References:
- Raubitzek et al., 2024, Physics Letters A 525, "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning"
"""

import numpy as np
import pandas as pd

n_samples = 1000  # Set the number of samples

# Generate the first feature x0
x0_class_0 = np.random.uniform(0, np.pi / 3, n_samples // 2)
x0_class_1 = np.random.uniform(2 * np.pi / 3, np.pi, n_samples // 2)

# Combine the x0 feature for both classes
x0 = np.concatenate((x0_class_0, x0_class_1))

# Generate the other features x1 and x2
x1 = np.zeros(n_samples)
x2 = np.random.uniform(0+0.00001, np.pi-0.00001, n_samples)

# Generate the class labels
y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

# Shuffle the dataset
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Create the final dataset
x0 = x0[indices]
x1 = x1[indices]
x2 = x2[indices]
y = y[indices]

# Create a DataFrame
df = pd.DataFrame({
    'x0': x0,
    'x1': x1,
    'x2': x2,
    'Class': y.astype(int)
})

# Save to CSV
file_name = f"synthetic_data_{n_samples}.csv"
df.to_csv(file_name, index=False)
print(f"Dataset saved to {file_name}")
