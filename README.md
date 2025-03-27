# LASSO Regression via Homotopy Method

This project implements a LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy method from first principles. The model is designed to efficiently compute a solution path of coefficients as the regularization parameter λ decreases, yielding sparse solutions that are especially useful when dealing with high-dimensional data or features exhibiting collinearity.

---

## Table of Contents

- [Overview](#overview)
- [Model Description and Use Cases](#model-description-and-use-cases)
- [Installation and Setup](#installation-and-setup)
- [Running the Code](#running-the-code)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Executing the Tests](#executing-the-tests)
  - [Usage Example](#usage-example)
  - [Generating Regression Data](#generating-regression-data)
- [Testing Methodology](#testing-methodology)
- [Exposed Parameters and Performance Tuning](#exposed-parameters-and-performance-tuning)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Overview

The LASSO model implemented here employs the Homotopy method to track the solution of a LASSO-regularized regression problem as the regularization parameter varies. This implementation is built from first principles using Numpy/Scipy, and does not rely on higher-level built-in models such as those in SciKit Learn (although SciKit Learn can be used for generating test data).

---

## Model Description and Use Cases

The **LassoHomotopyModel**:
- **What It Does:**  
  It computes a regularization path of coefficients for varying λ values, effectively transitioning from a highly regularized (sparse) solution to one that approximates ordinary least squares as regularization diminishes.
- **When to Use It:**  
  This model is particularly useful for:
  - **Feature Selection:** Automatically discarding irrelevant or collinear features.
  - **High-Dimensional Data:** Handling scenarios where the number of features is high.
  - **Preventing Overfitting:** Introducing sparsity to reduce model complexity.
  - **Interpretable Models:** Providing insight into feature importance through sparse coefficients.

---

## Installation and Setup

### Virtual Environment Setup

To ensure a clean environment and avoid dependency conflicts, please follow these steps:

1. **Activate the Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Deactivate the Virtual Environment When Done:**
   ```bash
   deactivate
   ```

---

## Running the Code

### Directory Structure

The project is organized as follows:
```
spring2025cs584-project1/
├── README.md
├── generate_regression_data.py
├── requirements.txt
└── LassoHomotopy/
    ├── model/
    │   └── LassoHomotopy.py
    └── tests/
        ├── __init__.py
        ├── collinear_data.csv
        ├── small_test.csv
        └── test_LassoHomotopy.py
```

### Executing the Tests

We use **PyTest** for ensuring the correctness and robustness of the implementation. To run all tests, navigate to the `tests` directory and run:

```bash
pytest
```

The tests include:
- **Initialization and Basic Regression:** Checks that the model initializes properly and can perform simple regression.
- **Sparsity Verification:** Ensures the model returns sparse solutions when presented with collinear data.
- **CSV Data Loading:** Validates functionality using datasets generated in CSV format.
- **Regularization Path Analysis:** Verifies that the coefficient paths change appropriately as λ decreases.

### Usage Example

Below is an example of how to fit the LASSO model and predict outcomes:

```python
import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Sample data
X = np.array([[1, 0], [0, 1], [1, 1], [2, 2]])
y = np.array([1, 2, 3, 4])

# Initialize and fit the model
model = LassoHomotopyModel(lambda_min=1e-6, n_lambdas=10)
results = model.fit(X, y)

# Predict using the fitted model
predictions = results.predict(X)
print("Predictions:", predictions)
```

### Generating Regression Data

The script `generate_regression_data.py` helps generate synthetic datasets with collinear features, which are useful for testing the model's feature selection capabilities. To run the script, execute:

```bash
python generate_regression_data.py
```

This will create two CSV files in the tests directory:
- `collinear_data.csv`: A larger dataset with collinear features.
- `small_test.csv`: A smaller dataset suitable for quick testing.

---

## Testing Methodology

To ensure the model works as intended, we have implemented comprehensive tests that cover:

- **Unit Testing:** Basic checks on initialization, coefficient path correctness, and prediction accuracy.
- **Sparsity Testing:** Verification that the model produces sparse solutions when input features are highly collinear.
- **Path Property Testing:** Ensuring that as the regularization parameter varies, the sparsity and coefficient values change as expected.
- **CSV-based Testing:** Running tests on datasets generated via CSV files to simulate real-world data ingestion.

---

## Exposed Parameters and Performance Tuning

The **LassoHomotopyModel** exposes several parameters for tuning:

- **`lambda_max`:** Maximum regularization parameter. If not provided, it is computed from the data.
- **`lambda_min`:** Minimum regularization parameter.
- **`n_lambdas`:** Number of lambda values along the regularization path.
- **`max_iter`:** Maximum iterations per lambda value to achieve convergence.
- **`tol`:** Tolerance for convergence to determine when iterative updates should stop.

These parameters allow the user to control the balance between computational efficiency and the precision of the solution.

---

## Limitations and Future Improvements

While the current implementation is robust for many regression tasks, there are a few limitations:

- **Numerical Stability:**  
  Despite adding a small regularization term for stability, extremely ill-conditioned data may still pose challenges.
- **Scalability:**  
  High-dimensional datasets may increase computation time. Optimizations such as parallelization could improve performance.
- **Active Set Management:**  
  The heuristic used for managing the active set is effective but may not be optimal in every scenario. Further research could refine this approach.

Given more time, these challenges could be mitigated with additional numerical techniques and algorithmic improvements.

---
Team Memebers-
Ananth Krishna Vasireddy - A20585441
Vishwashree Channaareddy Hanumanthareddy - A20556543
Aishwarya Ainala - A20546437
Yasaswini kakumani - A20547678
