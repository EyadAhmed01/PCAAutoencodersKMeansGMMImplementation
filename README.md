# Unsupervised Learning Algorithms - Implementation from Scratch

This project implements four fundamental unsupervised learning algorithms from scratch using only NumPy:

1. **Principal Component Analysis (PCA)**
2. **Autoencoders**
3. **K-Means Clustering**
4. **Gaussian Mixture Models (GMM)**

## Overview

This is a comprehensive implementation and evaluation of unsupervised learning techniques applied to the **Breast Cancer Wisconsin (Diagnostic) Dataset**. All algorithms are implemented from scratch without using machine learning libraries like scikit-learn, providing deep insight into how these algorithms work internally.

## Features

### Implemented Algorithms

- **PCA**: Dimensionality reduction using eigenvalue decomposition of the covariance matrix
- **Autoencoder**: Neural network-based dimensionality reduction with encoder-decoder architecture
- **K-Means**: Partition-based clustering with k-means++ initialization
- **GMM**: Probabilistic clustering using Expectation-Maximization algorithm

### Key Capabilities

- ✅ Pure NumPy implementations (no scikit-learn dependencies for core algorithms)
- ✅ Comprehensive experiments comparing different approaches
- ✅ Multiple evaluation metrics (Silhouette Score, ARI, NMI, Purity, etc.)
- ✅ Statistical analysis and hypothesis testing
- ✅ Extensive visualizations and analysis
- ✅ Computational complexity analysis

## Project Structure

```
.
├── main.ipynb          # Main notebook with all implementations and experiments
├── README.md           # This file
└── data.csv            # Breast Cancer Wisconsin Dataset (required)
```

## Requirements

The project requires the following Python packages:

- `numpy` - For numerical computations
- `pandas` - For data loading and manipulation
- `matplotlib` - For visualizations
- `scikit-learn` - For evaluation metrics and comparison (optional, used for metrics only)

Install dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains:
- **569 samples** with **30 features**
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, etc.
- Binary classification labels (Malignant/Benign) - used only for evaluation, not training

**Note**: You'll need to download `data.csv` and place it in the project root directory.

## Usage

1. **Open the notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run all cells** to execute the complete pipeline:
   - Data loading and preprocessing
   - Algorithm implementations
   - Comprehensive experiments
   - Evaluation and visualization

## Implementation Details

### Part 1: Core Implementations

Each algorithm is implemented as a class with the following methods:

- **PCA**: `fit()`, `transform()`, `inverse_transform()`, `reconstruction_error()`
- **Autoencoder**: `fit()`, `encode()`, `decode()`, `transform()`, `reconstruction_error()`
- **K-Means**: `fit()`, `predict()`, with k-means++ initialization
- **GMM**: `fit()`, `predict()`, `predict_proba()`, `bic()`, `aic()`

### Part 2: Comprehensive Experiments

Six main experiments are conducted:

1. **K-Means on Original Data**
2. **GMM on Original Data**
3. **K-Means after PCA**
4. **GMM after PCA**
5. **K-Means after Autoencoder**
6. **GMM after Autoencoder**

Each experiment includes:
- Optimal cluster number selection (Elbow method, BIC/AIC)
- Performance evaluation using multiple metrics
- Visualization of results

### Part 3: Evaluation and Analysis

Comprehensive evaluation includes:

- **Clustering Metrics**: Silhouette Score, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Purity, Davies-Bouldin Index, Calinski-Harabasz Index
- **Dimensionality Reduction Metrics**: Reconstruction error, explained variance
- **Visualizations**: 2D projections, elbow curves, BIC/AIC curves, training curves, cluster assignments
- **Statistical Analysis**: Paired t-tests, computational complexity analysis

## Key Results

The notebook provides detailed analysis comparing:
- Performance of clustering algorithms on original vs. reduced-dimensional data
- Effectiveness of PCA vs. Autoencoders for dimensionality reduction
- Optimal number of clusters for each method
- Trade-offs between different approaches

## Educational Value

This project is ideal for:
- Understanding the inner workings of unsupervised learning algorithms
- Learning how to implement machine learning algorithms from scratch
- Comparing different dimensionality reduction techniques
- Evaluating clustering performance using multiple metrics
- Statistical analysis of machine learning results

## Notes

- All implementations use only NumPy for core computations
- Random seed is set to 42 for reproducibility
- The dataset labels are used only for evaluation purposes (unsupervised learning)
- Comprehensive documentation and comments are included throughout the code

## License

This project is for educational purposes.

## Author

Assignment 4: Unsupervised Learning - Part 1
