# Credit Risk & Loan Default Prediction System

An end-to-end machine learning project that predicts the likelihood of loan default using applicant financial, employment, loan, and credit-history information.

## Project Overview

Loan default prediction is an important application of machine learning in the financial sector. The objective of this project is to build a binary classification system that estimates whether a loan is likely to default.

The project covers the complete machine learning workflow:

* Data cleaning and exploration
* Handling missing values
* Numerical and categorical preprocessing
* Feature encoding and scaling
* Class-imbalance handling
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Feature importance analysis
* Model serialization
* Streamlit deployment

## Dataset

The project uses the **Credit Risk Dataset** containing approximately 32,000 loan records.

### Target Variable

`loan_status`

* `0` → No Default
* `1` → Default

### Main Features

* Person age
* Person income
* Home ownership
* Employment length
* Loan intent
* Loan grade
* Loan amount
* Loan interest rate
* Loan percent of income
* Previous default history
* Credit history length

## Machine Learning Workflow

```text
Credit Risk Dataset
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
Missing Value Handling
        ↓
Numerical Scaling
        ↓
Categorical Encoding
        ↓
Model Training
        ↓
Model Comparison
        ↓
Cross-Validation
        ↓
XGBoost Hyperparameter Tuning
        ↓
Feature Importance
        ↓
Final Model
        ↓
Streamlit Deployment
```

## Models Evaluated

### 1. Logistic Regression

Used as the baseline classification model.

Default-class F1-score:

**0.66**

### 2. Random Forest

Used to capture nonlinear relationships and handle class imbalance.

Default-class F1-score:

**0.82**

### 3. XGBoost

Used as the main gradient-boosting model.

Default-class F1-score:

**0.83**

### 4. Tuned XGBoost

Hyperparameters were optimized using `RandomizedSearchCV`.

Best parameters:

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 1.0
```

Final test-set results:

| Metric            | Score |
| ----------------- | ----: |
| Accuracy          |  ~94% |
| Default Precision |  0.97 |
| Default Recall    |  0.74 |
| Default F1-score  |  0.84 |
| Macro F1-score    |  0.90 |

## Feature Importance

The final XGBoost model identified several influential features, including:

* Home ownership
* Loan percent of income
* Loan grade
* Loan intent
* Loan interest rate
* Person income

Feature importance was used to understand which processed input features contributed most to the model's predictions.

## Model Explainability & Data Considerations

The dataset contains features such as `loan_grade` and `loan_int_rate` that may reflect information generated
