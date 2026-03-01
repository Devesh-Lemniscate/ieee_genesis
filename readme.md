# IEEE SB Machine Learning Challenge: Fault Classification

## Project Overview
This repository contains our team's submission for the IEEE SB Machine Learning Challenge. The objective is to construct a binary classification model capable of detecting faulty devices based on 47 anonymized sensor readings (Features `F01` through `F47`).

## Methodology and Architecture
To achieve high generalization and prevent overfitting on tabular data, our solution utilizes a heavily regularized, tree-based ensemble approach. The pipeline is structured into four main phases:

### 1. Data Preprocessing
* **Deduplication:** Identified and removed 738 duplicate records from the training set prior to splitting. This prevents data leakage during cross-validation and ensures validation metrics accurately reflect unseen data performance.

### 2. Feature Engineering
We expanded the feature space from 47 to 60 variables to capture system-wide states rather than isolated sensor anomalies. 
* **Row-Wise Statistical States:** Calculated the `mean`, `standard deviation`, `min`, `max`, `skewness`, and `kurtosis` across all 47 sensors for each individual timestamp.
* **Sparsity Indicators:** Counted the exact number of sensors returning `0.0` per row, acting as a quantitative marker for systemic failure.
* **Unsupervised Manifold Learning (PCA):** Applied Principal Component Analysis (PCA) to compress the 47 sensors into 5 macro-features, forcing the model to evaluate the overall variance of the system.
* **State Clustering (K-Means):** Grouped the combined training and test datasets into 5 distinct operational states using K-Means. We extracted the distance to the nearest cluster center as a spatial anomaly detection feature.

### 3. Modeling: 3-Way Stacked Ensemble
The core predictive engine is an equally weighted blend of three gradient-boosted decision tree frameworks. All models utilize early stopping to halt training at peak generalization.
* **XGBoost:** Configured for conservative depth (`max_depth=6`) with heavy column and row subsampling (`0.8`).
* **LightGBM:** Optimized for leaf-wise growth (`num_leaves=31`) to handle the engineered feature space efficiently.
* **CatBoost:** Utilized for its robust handling of numerical distributions and resistance to overfitting via symmetric trees.

### 4. Validation and Threshold Optimization
* **Validation Strategy:** Evaluated using strict Stratified 5-Fold Cross-Validation (`random_state=42`) to maintain the exact 60/40 target class balance across all isolated folds.
* **Probability Thresholding:** Instead of a default 0.50 cutoff, the ensemble's Out-Of-Fold (OOF) probabilities were systematically scanned. The optimal decision boundary was mathematically determined to be `0.40`, maximizing target capture accuracy given the dataset's slight imbalance.

## Final Model Performance
* **Out-Of-Fold (OOF) Accuracy:** 98.82%
* **OOF ROC-AUC:** 0.9989

## Setup and Execution

### Prerequisites
Ensure Python 3.8+ is installed on your system. Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Devesh-Lemniscate/ieee_genesis.git](https://github.com/Devesh-Lemniscate/ieee_genesis.git)
cd ieee_genesis
pip install -r requirements.txt
```

### Dataset Placement
Download the competition files (`TRAIN.csv` and `TEST.csv`) and place them directly in the root directory of the cloned repository.

### Usage
1. Launch the Jupyter environment:
   ```bash
   jupyter notebook
   ```
2. Open the `.ipynb` notebook file.
3. Execute all cells sequentially from top to bottom. 
4. The pipeline will automatically process the data, engineer the features, train the 5-fold ensemble, calculate the optimal threshold, and generate the final predictions in a file named `FINAL_ULTIMATE.csv`.

