"""
Symmetry-Based Feature Maps for Quantum-Inspired Machine Learning

Author: Dr. techn Sebastian Raubitzek MSc. BSc.

The `SymmetryFeatureMaps` class implements symmetry group-based feature maps for quantum-inspired kernel matrix computation. By applying transformations based on Lie groups, such as SO(n), SL(n), SU(n), GL(n), U(n), O(n), and T(n), the class encodes input data into higher-dimensional feature spaces that capture the inherent symmetries of the data.

These feature maps are used for transforming input data vectors into representations based on the structure of symmetry groups, which are fundamental in quantum mechanics and quantum computing.

### Key Components:
1. **Symmetry Groups**:
   - The class supports multiple symmetry groups:
     - **SO(n)**: Special Orthogonal group, representing rotations.
     - **SL(n)**: Special Linear group.
     - **SU(n)**: Special Unitary group, commonly used in quantum mechanics.
     - **GL(n)**: General Linear group.
     - **U(n)**: Unitary group.
     - **O(n)**: Orthogonal group.
     - **T(n)**: Translation group.
   - Each symmetry group is associated with a number of generators, which are mathematical objects that define the transformations in the group.

2. **Generators**:
   - For each symmetry group, the class generates the appropriate Lie algebra generators using helper functions from the `func_sun` module. These generators are then used to transform the input data based on the group type.

3. **Group Sizes**:
   - The class computes the minimum number of group generators required to match the number of features in the input data. This ensures that the feature map can accommodate the input dimension.

4. **Feature Map Transformation**:
   - The class applies a quantum-inspired feature map by using the group generators of the selected symmetry group to create a matrix representation of the group element. This matrix is applied to the input data to transform it into a new feature vector, which can be real or complex depending on the configuration.

5. **Real and Complex Representations**:
   - The feature maps can output either real or complex vectors. If the output needs to be real, the class converts the complex vector into a real-valued vector by separating the real and imaginary parts.

### Methods:
- **`get_group_sizes(self)`**:
   - Returns a dictionary containing the number of features and the size (number of generators) for each symmetry group.

- **`apply_feature_map(self, X, group_type, output_real=False, return_group_n=False)`**:
   - Applies the specified symmetry group-based feature map to an input vector `X`. The `group_type` specifies which symmetry group to use (e.g., 'SO', 'SL', 'SU', 'GL', 'U', 'O', 'T').
   - Parameters:
     - `X` (numpy.ndarray): A vector of real numbers to be transformed.
     - `group_type` (str): The name of the symmetry group to apply ('SO', 'SL', 'SU', 'GL', 'U', 'O', 'T').
     - `output_real` (bool): Whether to convert the complex output vector to a real-valued vector.
     - `return_group_n` (bool): Whether to return the size of the group along with the transformed feature vector.
   - Returns:
     - A transformed feature vector (and optionally the group size).

- **`generic_feature_map(self, X, generators, output_real=False)`**:
   - A generic method for applying a feature map using a set of generators for any symmetry group.
   - Parameters:
     - `X` (numpy.ndarray): The input data vector.
     - `generators` (list): The list of group generators for the chosen symmetry group.
     - `output_real` (bool): Whether to convert the complex output vector to a real-valued vector.
   - Returns:
     - A transformed feature vector.

- **`complex_to_real_vector(self, vector)`**:
   - Converts a complex vector into a real-valued vector by concatenating the real and imaginary parts.
   - Parameters:
     - `vector` (numpy.ndarray): A complex vector.
   - Returns:
     - A real-valued vector with twice the number of components as the input complex vector.

### Group-Specific Feature Maps:
- **`SOn_feature_map(self, X, output_real=False)`**:
   - Applies the SO(n) symmetry group-based feature map to the input vector `X`.

- **`SLn_feature_map(self, X, output_real=False)`**:
   - Applies the SL(n) symmetry group-based feature map to the input vector `X`.

- **`SUn_feature_map(self, X, output_real=False)`**:
   - Applies the SU(n) symmetry group-based feature map to the input vector `X`.

- **`GLn_feature_map(self, X, output_real=False)`**:
   - Applies the GL(n) symmetry group-based feature map to the input vector `X`.

- **`Un_feature_map(self, X, output_real=False)`**:
   - Applies the U(n) symmetry group-based feature map to the input vector `X`.

- **`On_feature_map(self, X, output_real=False)`**:
   - Applies the O(n) symmetry group-based feature map to the input vector `X`.

- **`Tn_feature_map(self, X, output_real=False)`**:
   - Applies the translation group T(n) feature map to the input vector `X`.

### Context and Purpose:
- This class is central to the quantum-inspired approach in machine learning, where symmetry groups are used to transform input data into high-dimensional feature spaces. The encoded symmetries provide a powerful method for representing data, allowing for enhanced learning capabilities in models such as quantum kernel estimators.
- The class is used to experiment with different symmetry groups and analyze how they affect model performance in tasks such as classification.
- This methodology is discussed in the paper "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning" (Raubitzek et al., 2024), where symmetry-based feature maps are applied to machine learning tasks.

### Notes:
- The feature maps in this class simulate the process of transforming classical data into quantum-like representations, leveraging the mathematical properties of Lie groups.
- The current implementation supports several well-known groups, but more groups can be added by extending the list of generators.

References:
- Raubitzek et al., 2024, Physics Letters A 525, "Quantum-inspired Kernel Matrices: Exploring Symmetry in Machine Learning"
"""

import numpy as np
from scipy.linalg import expm
import func_sun

class SymmetryFeatureMaps:
    def __init__(self, num_features):
        self.num_features = num_features
        # Initialize the group sizes
        size_SO = find_so_group(num_features)
        size_SL = find_sl_group(num_features)  # SL and SU have the same size
        size_SU = find_su_group(num_features)  # SL and SU have the same size
        size_GL = find_gl_u_group(num_features)
        size_U = find_gl_u_group(num_features)
        size_O = find_o_group(num_features)
        size_T = find_translation_group(num_features)

        self.size_SO = size_SO
        self.size_SL = size_SL
        self.size_SU = size_SU
        self.size_GL = size_GL
        self.size_U = size_U
        self.size_O = size_O
        self.size_T = size_T

        # Generate the corresponding generators for each group
        self.group_generators_SO = func_sun.generate_SO(size_SO)
        self.group_generators_SL = func_sun.generate_SL_from_SU(size_SL)
        self.group_generators_GL = func_sun.generate_GL(size_GL)
        self.group_generators_O = func_sun.generate_O(size_O)
        self.group_generators_U = func_sun.generate_U(size_U)
        self.group_generators_SU = func_sun.generate_SU(size_SU)
        self.group_generators_T = func_sun.generate_T(size_T)

    def get_group_sizes(self):
        """
        Returns a dictionary containing the number of features and the number of generators for each group.
        """
        group_sizes = {
            "num_features": self.num_features,
            "size_SO": len(self.group_generators_SO),
            "size_SL": len(self.group_generators_SL),
            "size_SU": len(self.group_generators_SU),
            "size_GL": len(self.group_generators_GL),
            "size_U": len(self.group_generators_U),
            "size_O": len(self.group_generators_O),
            "size_T": len(self.group_generators_T)
        }
        return group_sizes

    def apply_feature_map(self, X, group_type, output_real=False, return_group_n=False):
        if group_type == "SO":
            if return_group_n: return self.SOn_feature_map(X, output_real=output_real), self.size_SO
            else: return self.SOn_feature_map(X, output_real=output_real)
        elif group_type == "SL":
            if return_group_n: return self.SLn_feature_map(X, output_real=output_real), self.size_SL
            else: return self.SLn_feature_map(X, output_real=output_real)
        elif group_type == "SU":
            if return_group_n: return self.SUn_feature_map(X, output_real=output_real), self.size_SU
            else: return self.SUn_feature_map(X, output_real=output_real)
        elif group_type == "GL":
            if return_group_n: return self.GLn_feature_map(X, output_real=output_real), self.size_GL
            else: return self.GLn_feature_map(X, output_real=output_real)
        elif group_type == "U":
            if return_group_n: return self.Un_feature_map(X, output_real=output_real), self.size_U
            else: return self.Un_feature_map(X, output_real=output_real)
        elif group_type == "O":
            if return_group_n: return self.On_feature_map(X, output_real=output_real), self.size_O
            else: return self.On_feature_map(X, output_real=output_real)
        elif group_type == "T":
            if return_group_n: return self.Tn_feature_map(X, output_real=output_real), self.size_T
            else: return self.Tn_feature_map(X, output_real=output_real)
        else:
            raise ValueError(f"Unknown group type: {group_type}")

    def SOn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SO, output_real=output_real)

    def SLn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SL, output_real=output_real)

    def SUn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SU, output_real=output_real)

    def GLn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_GL, output_real=output_real)

    def Un_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_U, output_real=output_real)

    def On_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_O, output_real=output_real)

    def Tn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_T, output_real=output_real)

    def complex_to_real_vector(self, vector):
        """
        Convert a complex vector with n components into a real vector with 2n components.
        """
        real_part = vector.real
        imag_part = vector.imag
        return np.concatenate((real_part, imag_part))

    def generic_feature_map(self, X, generators, output_real=False):
        num_features = len(X)
        num_generators = len(generators)

        group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)

        group_element = expm(1j * group_element)

        dim = generators[0].shape[0]
        initial_vector = np.ones(dim) / np.sqrt(dim)

        transformed_vector = np.dot(group_element, initial_vector)

        # Convert to real vector if required
        if output_real:
            transformed_vector = self.complex_to_real_vector(transformed_vector)

        return transformed_vector

def find_su_group(num_features):
    """
    Find the smallest n for SU(n) that provides enough generators for the number of features.
    """
    n = 2  # Start from SU(2)
    while True:
        num_generators = n**2 - 1
        if num_generators >= num_features:
            # Ensure at least equal number of generators as features
            return max(n, int(np.ceil(np.sqrt(num_features + 1))))
        n += 1

def find_sl_group(num_features):
    return find_su_group(num_features)  # Same as SU(n)

def find_so_group(num_features):
    n = 1  # Start from SO(1)
    while True:
        num_generators = n * (n - 1) // 2
        if num_generators >= num_features:
            return n
        n += 1

def find_gl_u_group(num_features):
    n = 1  # Start from GL(1) or U(1)
    while True:
        num_generators = n**2
        if num_generators >= num_features:
            return n
        n += 1

def find_u_group(num_features):
    n = 1  # Start from GL(1) or U(1)
    while True:
        num_generators = n**2
        if num_generators >= num_features:
            return n
        n += 1

def find_o_group(num_features):
    return find_so_group(num_features)  # Same as SO(n)

def find_translation_group(num_features):
    """
    Find the smallest n for the translation group T(n) that provides enough generators for the number of features.
    In the case of the translation group, each dimension has a single generator.
    """
    n = 1  # Start from T(1)
    while True:
        num_generators = n  # Each dimension has one generator
        if num_generators >= num_features:
            return n
        n += 1
