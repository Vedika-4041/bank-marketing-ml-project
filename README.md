# Machine Learning Assignment - 2

**Student Name:** Vedika Patil

**BITS ID:** 2024DC04014

**Program:** M.Tech (AIML/DSE)

**Institution:** BITS Pilani (WILP)

---

## Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on historical marketing campaign data from a Portuguese banking institution. This is a binary classification problem where the target variable is whether the client subscribed to a term deposit (yes/no).

The banking institution wants to improve the effectiveness of their marketing campaigns by identifying potential customers who are more likely to subscribe to term deposits. By building and comparing multiple machine learning classification models, we aim to:

1. Predict customer subscription behavior with high accuracy
2. Identify key features that influence subscription decisions
3. Compare different classification algorithms to determine the most effective model
4. Provide actionable insights for optimizing future marketing campaigns

This solution will help the bank reduce marketing costs by targeting high-probability customers and improve conversion rates through data-driven decision making.

---

## Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Source:** Kaggle

**Link:** https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

### Dataset Overview
- **Total Samples:** 11,163 rows
- **Total Features:** 17 columns (16 input features + 1 target variable)
- **Target Variable:** deposit (binary: yes/no)
- **Class Distribution:** Approximately balanced (52.6% no, 47.4% yes)
- **Missing Values:** None
- **File Size:** 898 KB

### Feature Description

**Bank Client Data:**
1. **age:** Age of the customer (numeric)
2. **job:** Type of job (categorical: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
3. **marital:** Marital status (categorical: divorced, married, single, unknown)
4. **education:** Education level (categorical: primary, secondary, tertiary, unknown)
5. **default:** Has credit in default? (categorical: yes, no, unknown)
6. **balance:** Average yearly balance in euros (numeric)
7. **housing:** Has housing loan? (categorical: yes, no, unknown)
8. **loan:** Has personal loan? (categorical: yes, no, unknown)

**Campaign Contact Data:**
9. **contact:** Contact communication type (categorical: cellular, telephone)

10. **day:** Last contact day of the month (numeric)

11. **month:** Last contact month of year (categorical: jan, feb, mar, ..., nov, dec)

12. **duration:** Last contact duration in seconds (numeric)

13. **campaign:** Number of contacts performed during this campaign (numeric)

14. **pdays:** Number of days since client was last contacted from previous campaign (numeric, 999 means not previously contacted)

15. **previous:** Number of contacts performed before this campaign (numeric)

16. **poutcome:** Outcome of previous marketing campaign (categorical: failure, nonexistent, success)

**Target Variable:**
17. **deposit:** Has the client subscribed to a term deposit? (binary: yes, no)

### Data Preprocessing
- Label encoding applied to all categorical variables
- Standard scaling applied for distance-based algorithms (Logistic Regression, KNN)
- Stratified train-test split (80-20) to maintain class balance
- 3-fold stratified cross-validation for model evaluation

---

## Models Used

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8470 | 0.9000 | 0.8150 | 0.8500 | 0.8320 | 0.6850 |
| Decision Tree | 0.8400 | 0.8750 | 0.8000 | 0.8450 | 0.8220 | 0.6700 |
| kNN | 0.8510 | 0.8900 | 0.8250 | 0.8400 | 0.8320 | 0.6950 |
| Naive Bayes | 0.8470 | 0.8034 | 0.8100 | 0.8600 | 0.8340 | 0.6850 |
| Random Forest (Ensemble) | 0.8421 | 0.9100 | 0.8200 | 0.8500 | 0.8350 | 0.6800 |
| XGBoost (Ensemble) | 0.8550 | 0.9200 | 0.8350 | 0.8600 | 0.8470 | 0.7050 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Demonstrates strong baseline performance with 84.70% accuracy and excellent AUC of 0.90. The model shows good balance between precision (81.50%) and recall (85.00%), making it suitable for interpretable predictions. Its linear decision boundary works well for this dataset, suggesting that the relationship between features and target is largely linear. The high AUC indicates strong discriminative power between classes. Training time is minimal, making it efficient for production deployment. |
| Decision Tree | Achieves 84.00% accuracy with moderate AUC of 0.8750. The model provides excellent interpretability through feature importance rankings, revealing that call duration is the most influential factor (46% importance). However, it shows slight overfitting tendencies compared to ensemble methods. Precision is relatively lower (80.00%), indicating more false positives. The model is useful for understanding feature relationships but may not be optimal for deployment due to instability with small data variations. |
| kNN | Performs well with 85.10% accuracy, second only to XGBoost. AUC of 0.89 is respectable, though lower than tree-based ensembles. The model achieves good precision (82.50%) and balanced recall (84.00%). Performance is sensitive to the choice of k parameter and distance metric. Requires feature scaling, which was applied. Computational cost is higher during prediction phase as it needs to compute distances to all training samples. Works well due to clear cluster patterns in the data. |
| Naive Bayes | Achieves 84.70% accuracy but has the lowest AUC (0.8034) among all models, indicating weaker class separation capability. Despite this, it maintains high recall (86.00%), making it effective at identifying potential subscribers. The model's assumption of feature independence is violated in this dataset (e.g., age and job are correlated), which limits performance. Training is extremely fast, and the model is useful when computational resources are constrained or quick baseline results are needed. |
| Random Forest (Ensemble) | Strong ensemble performance with 84.21% accuracy and high AUC of 0.91. Combines multiple decision trees to reduce overfitting and improve generalization. Feature importance analysis provides insights into key predictors while maintaining robustness. Precision (82.00%) and recall (85.00%) are well-balanced. The model handles non-linear relationships effectively and is resistant to outliers. Training time is moderate, and the model requires minimal hyperparameter tuning to achieve good results. |
| XGBoost (Ensemble) | Best overall performance across all metrics: 85.50% accuracy, 0.92 AUC, 84.70% F1 score, and highest MCC (0.7050). The gradient boosting approach iteratively corrects errors from previous trees, resulting in superior predictive power. Excels at handling imbalanced data and captures complex feature interactions. Precision (83.50%) and recall (86.00%) are both highest among all models. Recommended for production deployment. Requires more computational resources and careful hyperparameter tuning but delivers optimal results for this marketing prediction task. |

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vedika-4041/bank-marketing-ml-project.git
cd bank-marketing-ml-project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Option 1: Train All Models

Ensure virtual environment is activated, then run:

```bash
python train_all_models.py
```

This script will:
- Load and preprocess the bank.csv dataset
- Train all 6 models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost)
- Calculate all evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Save trained models as .pkl files in the models/ directory
- Display performance comparison

**Expected Runtime:** 2-3 minutes (depending on system configuration)

### Option 2: Train Individual Models

Ensure virtual environment is activated, then run a specific model:

```bash
python models/logistic_regression.py
python models/decision_tree.py
python models/knn.py
python models/naive_bayes.py
python models/random_forest.py
python models/xgboost_model.py
```

### Option 3: Run Streamlit Web Application

Ensure virtual environment is activated, then launch the application:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at http://localhost:8501

**Application Features:**
- Upload test dataset (CSV format)
- Select any of the 6 trained models
- View prediction results
- Display all evaluation metrics
- Show confusion matrix and classification report
- Download predictions as CSV file

---

## Project Structure

```
term-deposit-opening-supervised-ml-project/
│
├── README.md                          # This file
├── bank.csv                           # Dataset
├── requirements.txt                   # Python dependencies
├── app.py                             # Streamlit web application
├── train_all_models.py                # Batch training script
│
└── models/                            # Model implementations
    ├── logistic_regression.py
    ├── logistic_regression_model.pkl
    ├── decision_tree.py
    ├── decision_tree_model.pkl
    ├── knn.py
    ├── knn_model.pkl
    ├── naive_bayes.py
    ├── naive_bayes_model.pkl
    ├── random_forest.py
    ├── random_forest_model.pkl
    ├── xgboost_model.py
    └── xgboost_model.pkl
```

---

## Troubleshooting

### Common Issues and Solutions

**Issue 1: ModuleNotFoundError**
```
Solution: Activate virtual environment and ensure all dependencies are installed
venv\Scripts\activate  (Windows)
source venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt
```

**Issue 2: FileNotFoundError for bank.csv**
```
Solution: Ensure bank.csv is in the project root directory
ls bank.csv
```

**Issue 3: Streamlit port already in use**
```
Solution: Use a different port
streamlit run app.py --server.port 8502
```

**Issue 4: Model training memory error**
```
Solution: Close other applications or train models individually
python models/logistic_regression.py
```

---

## References

1. UCI Machine Learning Repository - Bank Marketing Dataset
   https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

2. Scikit-learn Documentation
   https://scikit-learn.org/stable/

3. XGBoost Documentation
   https://xgboost.readthedocs.io/

4. Streamlit Documentation
   https://docs.streamlit.io/

---

**Submitted by:** Vedika Patil (2024DC04014)
**Course:** Machine Learning - Assignment 2
**Program:** M.Tech (AIML/DSE)
**Institution:** BITS Pilani (WILP)
**Date:** January 2026
