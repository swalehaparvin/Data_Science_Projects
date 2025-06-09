# Supervised Machine Learning Tutorial

A comprehensive hands-on guide to supervised machine learning techniques including linear regression, k-nearest neighbors, cross-validation, and regularized regression models.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial Contents](#tutorial-contents)
- [Usage](#usage)
- [Key Concepts](#key-concepts)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains practical implementations and examples of fundamental supervised machine learning algorithms. The tutorial progresses from basic linear regression to advanced regularization techniques, providing both theoretical understanding and hands-on coding experience.

### What You'll Learn
- Linear regression for housing price prediction
- K-Nearest Neighbors (KNN) algorithm from scratch
- Cross-validation for robust model evaluation
- Ridge and Lasso regression for regularization
- Feature selection and overfitting prevention

## 🔧 Prerequisites

- Basic understanding of Python programming
- Familiarity with mathematical concepts (linear algebra basics)
- Understanding of statistics fundamentals
- No prior machine learning experience required!

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/supervised-ml-tutorial.git
cd supervised-ml-tutorial

# Install required packages
pip install numpy pandas matplotlib scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from collections import Counter
```

## 📚 Tutorial Contents

### 1. Linear Regression for Housing Prices
- **File**: `01_linear_regression.py`
- **Concept**: Basic linear regression using f(x) = w·x + b
- **Example**: Predicting house prices based on size
- **Output**: $340k prediction for 1200 sqft house

```python
# Example usage
w = 200  # weight
b = 100  # bias
x_i = 1.2  # house size (1200 sqft)
price = w * x_i + b  # $340k
```

### 2. K-Nearest Neighbors (KNN) Implementation
- **File**: `02_knn_algorithm.py`
- **Concept**: Classification based on nearest neighbors
- **Features**: Custom Euclidean distance calculation
- **Example**: Classifying points into categories A or B

```python
# Custom KNN implementation
def knn_predict(training_data, training_labels, test_point, k):
    # Calculate distances, sort, and predict
    # Returns most common label among k nearest neighbors
```

### 3. Cross-Validation for Model Evaluation
- **File**: `03_cross_validation.py`
- **Concept**: 6-fold cross-validation for robust evaluation
- **Purpose**: Prevent overfitting and get reliable performance metrics
- **Output**: Individual scores, mean, std, and confidence intervals

```python
# 6-fold cross-validation
kf = KFold(n_splits=6, shuffle=True, random_state=5)
cv_scores = cross_val_score(reg, X, y, cv=kf)
```

### 4. Ridge Regression (L2 Regularization)
- **File**: `04_ridge_regression.py`
- **Concept**: Penalizes large coefficients to prevent overfitting
- **Alpha range**: 0.1 to 10,000
- **Behavior**: Shrinks coefficients but keeps all features

```python
# Test multiple alpha values
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    # Fit and evaluate
```

### 5. Lasso Regression (L1 Regularization)
- **File**: `05_lasso_regression.py`
- **Concept**: Automatic feature selection through L1 penalty
- **Behavior**: Sets irrelevant feature coefficients to zero
- **Advantage**: Produces sparse, interpretable models

## 🚀 Usage

### Quick Start
```bash
# Run individual tutorials
python 01_linear_regression.py
python 02_knn_algorithm.py
python 03_cross_validation.py
python 04_ridge_regression.py
python 05_lasso_regression.py

# Or run the complete tutorial
python complete_tutorial.py
```

### Interactive Notebook
```bash
# Launch Jupyter notebook
jupyter notebook supervised_ml_tutorial.ipynb

# Or use Google Colab
# Upload the .ipynb file to Google Colab
```

## 🧠 Key Concepts

### Linear Regression
- **Formula**: f(x) = w·x + b
- **Use case**: Continuous target prediction
- **Example**: Housing price prediction

### K-Nearest Neighbors
- **Principle**: "Similar inputs produce similar outputs"
- **Distance metric**: Euclidean distance
- **Parameter**: k (number of neighbors)

### Cross-Validation
- **Purpose**: Robust model evaluation
- **Method**: Split data into k folds
- **Benefit**: Uses all data for both training and testing

### Regularization
| Method | Penalty | Effect | Use Case |
|--------|---------|---------|----------|
| Ridge (L2) | Sum of squared coefficients | Shrinks coefficients | When all features matter |
| Lasso (L1) | Sum of absolute coefficients | Feature selection | When some features are irrelevant |

## 📊 Results

### Expected Outputs

**Linear Regression:**
- Housing price prediction: $340k for 1200 sqft
- Clear linear relationship visualization

**KNN:**
- Classification accuracy on test data
- Visualization of decision boundaries

**Cross-Validation:**
- 6 individual fold scores
- Mean R² score with confidence intervals
- Standard deviation of performance

**Ridge Regression:**
- Performance scores across different alpha values
- Coefficient shrinkage demonstration

**Lasso Regression:**
- Automatic feature selection results
- Sparse coefficient vectors
- Feature importance ranking

## 📈 Performance Metrics

- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **Cross-validation scores**: Consistency across folds
- **Feature selection**: Number of non-zero coefficients

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -am 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Create a Pull Request

### Contribution Ideas
- Add more algorithms (SVM, Decision Trees, etc.)
- Implement additional regularization techniques
- Add more visualization examples
- Create advanced feature engineering examples

## 📚 Further Learning

### Recommended Next Steps
1. **Ensemble Methods**: Random Forest, Gradient Boosting
2. **Feature Engineering**: Polynomial features, feature scaling
3. **Model Selection**: Grid search, random search
4. **Advanced Regularization**: Elastic Net, Group Lasso

### Additional Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML library
- Housing dataset contributors
- Machine learning community for best practices

---

**Happy Learning! 🚀**

*If you found this tutorial helpful, please ⭐ star the repository and share it with others!*
