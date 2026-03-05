# IEEE SB Machine Learning Challenge: Fault Classification

## Project Overview
This repository contains our team's submission for the IEEE SB Machine Learning Challenge. The task was to make a binary classification model capable of detecting faulty devices based on 47 sensor readings (Features `F01` through `F47`).

## Methodology and Architecture
To  prevent overfitting on tabular data we used a regularized tree based ensemble approach. The pipeline is structured into four main phases:

### 1. Data Preprocessing
* **Deduplication:** Identified and removed 738 duplicate records from the training set before splitting. This prevented data leakage during cross-validation and ensured that validation metrics accurately reflect unseen data performance.

### 2. Feature Engineering
We expanded the feature space from 47 to 60 variables to capture system wide states rather than only sensor anomalies. 
* **Row Wise Statistical States:** Calculated the `mean`, `standard deviation`, `min`, `max`, `skewness`, and `kurtosis` across all 47 sensors for each individual timestamp.
* **Sparsity Indicators:** Counted the exact number of sensors returning `0.0` per row depicting system failure.
* **Unsupervised Manifold Learning (PCA):** Applied Principal Component Analysis (PCA) to compress the 47 sensors into 5 macro features forcing the model to evaluate the overall variance of the system.
* **State Clustering (K-Means):** Used K-Means to group the training and testing dataset into 5 states. Then extracted the distance to the nearest cluster center.

### 3. Modeling: 3-Way Stacked Ensemble
We used 3 gradient boosted decision tree models with each having equal weight as the core predictive engine. All models uses early stopping to prevent overfitting. The 3 models were :- 
* **XGBoost:**
* **LightGBM:**
* **CatBoost:**

### 4. Validation and Threshold Optimization
* **Cross Validation:** Evaluated using 5 Fold Cross Validation (`random_state=42`) and maintained the exact 60/40 ratio of working and faulty devices across all isolated folds.
* **Probability Thresholding:** Instead of flagging the device at `50%` probability, we lowered the trigger and found that the flagging the device as faulty at `40%` probability found the maximum faulty devices. 

## Final Model Performance
* **Out-Of-Fold (OOF) Accuracy:** 98.82%
* **OOF ROC-AUC:** 0.9989

## Setup and Execution

### Prerequisites
Ensure Python 3.8+ is installed on the system. Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Devesh-Lemniscate/ieee_genesis.git](https://github.com/Devesh-Lemniscate/ieee_genesis.git)
cd ieee_genesis
pip install -r requirements.txt
```

### Dataset Placement
Download the competition ML Challenge Dataset folder and inside it will be (`TRAIN.csv` and `TEST.csv`) files.

### Usage
1. Launch the Jupyter environment:
   ```bash
   jupyter notebook
   ```
2. Open the `.ipynb` notebook file.
3. Execute all cells sequentially from top to bottom. 
4. The pipeline will automatically process the data, engineer the features, train the 5-fold ensemble, calculate the optimal threshold, and generate the final predictions in a file named `FINAL.csv`.

