# Credit Card Fraud Detection

This script aims to evaluate the performance of different machine learning models on a credit card fraud detection task. The dataset used is assumed to be named `Creditcard_data.csv`, and it should contain features and labels, where 'Class' is the binary label indicating whether a transaction is fraudulent (1) or not (0).

## Prerequisites

- Python 3.x
- Install required libraries using:

    ```bash
    pip install pandas imbalanced-learn scikit-learn xgboost
    ```

## Code Overview

### Data Handling

The code uses the `pandas` library to handle the dataset. It loads the data into a DataFrame and checks for the balance of the target variable 'Class'.

### Sampling Techniques

The script employs different sampling techniques to handle imbalanced data. It uses the `imbalanced-learn` library for Random Under-Sampling (RUS), Random Over-Sampling (ROS), Tomek Links (TL), SMOTE (Synthetic Minority Oversampling Technique), and NearMiss (NM).

### Machine Learning Models

The following machine learning models are utilized:
- Logistic Regression (`LogReg`)
- Random Forest Classifier (`RandFor`)
- Support Vector Classifier (`SVC`)
- k-Nearest Neighbors Classifier (`KNN`)
- XGBoost Classifier (`XGBC`)

### Model Evaluation

The code defines functions to build models, calculate performance metrics (recall and accuracy), and evaluate models based on different sampling techniques. The performance results are stored in DataFrames.

### Results and Output

The script prints and saves the recall and accuracy results for each model and sampling technique to CSV files (`recall_results_modified.csv` and `accuracy_results_modified.csv`).

## How to Use

1. Ensure your dataset is named `Creditcard_data.csv` and contains the necessary features and labels.
2. Install the required dependencies.
3. Run the script.

```bash
python your_script_name.py
