# Logistic Regression — Admittance & Bank Subscription

An educational project demonstrating **Logistic Regression** using the `statsmodels` library across multiple labs and practice exercises. The project is split into two themes: predicting **university admittance** based on SAT scores and **bank term-deposit subscription** based on client interaction data.

## Overview

Logistic regression is a fundamental classification algorithm that models the probability of a binary outcome using the logistic (sigmoid) function. Unlike linear regression, the output is constrained between 0 and 1, making it ideal for yes/no classification tasks.

This project progressively builds understanding through **two guided labs** and **four practice exercises**, each in its own directory.

## Project Structure

```
logistic_regression/
├── lab1/
│   ├── logistic_regression.ipynb       # Simple logistic regression (SAT → Admitted)
│   └── 2.01.+Admittance.csv           # Admittance dataset (168 rows)
├── lab2/
│   ├── binary_predictors.ipynb         # Logistic regression with binary predictors
│   ├── 2.02.+Binary+predictors.csv     # Admittance dataset with Gender (168 rows)
│   └── 2.03.+Test+dataset.csv          # Held-out test set (19 rows)
├── practice1/
│   ├── log_reg.ipynb                   # Exercise: simple logistic regression on bank data
│   └── Bank_data.csv                   # Bank marketing dataset (518 rows)
├── practice2/
│   ├── Binary+Predictors+in+a+Logistic+Regression+-+Exercise.ipynb      # Exercise (starter)
│   ├── Binary+Predictors+in+a+Logistic+Regression+-+Exercise (1).ipynb  # Exercise (solved)
│   └── Bank_data.csv                   # Bank marketing dataset (518 rows)
├── practice3/
│   ├── Calculating+the+Accuracy+of+the+Model+-+Exercise.ipynb  # Exercise: accuracy & confusion matrix
│   └── Bank_data.csv                   # Bank marketing dataset (518 rows)
├── practice4/
│   ├── Bank_data.csv                   # Bank marketing dataset (518 rows)
│   └── Bank_data_testing.csv           # Held-out test set (222 rows)
├── requirements.txt
└── README.md                           # This file
```

## Datasets

### Admittance Dataset (Labs 1 & 2)

| Property | Value |
|---|---|
| **File** | `2.01.+Admittance.csv` / `2.02.+Binary+predictors.csv` |
| **Samples** | 168 |
| **Features** | `SAT` (continuous); `Gender` (categorical — lab 2 only) |
| **Target** | `Admitted` — `Yes` or `No` |
| **Missing values** | None |

### Bank Marketing Dataset (Practices 1–4)

| Property | Value |
|---|---|
| **File** | `Bank_data.csv` |
| **Samples** | 518 |
| **Features** | `interest_rate`, `credit`, `march`, `may`, `previous`, `duration` |
| **Target** | `y` — `yes` (subscribed) or `no` (did not subscribe) |
| **Missing values** | None |

A separate test set (`Bank_data_testing.csv`, 222 rows) is provided in `practice4/` for model evaluation on unseen data.

## Lab & Practice Workflows

### Lab 1 — Simple Logistic Regression

**Notebook:** [`lab1/logistic_regression.ipynb`](lab1/logistic_regression.ipynb)

Introduces logistic regression with a single continuous predictor.

1. Loads the admittance dataset (`SAT`, `Admitted`).
2. Maps the target to binary (Yes → 1, No → 0).
3. Visualizes the relationship with a **scatter plot** (SAT vs. Admitted).
4. Adds a constant and fits a **Logit model** via `statsmodels` (MLE).
5. Interprets the **regression summary** — coefficients, p-values, Pseudo R², Log-Likelihood.

### Lab 2 — Binary Predictors & Model Testing

**Notebook:** [`lab2/binary_predictors.ipynb`](lab2/binary_predictors.ipynb)

Extends the model with a binary categorical predictor and introduces model testing.

1. Loads the extended admittance dataset (`SAT`, `Admitted`, `Gender`).
2. Maps categorical columns to binary (Yes/No → 1/0, Female/Male → 1/0).
3. Fits a **multivariate Logit model** with both `SAT` and `Gender` as predictors.
4. Examines the regression summary — Gender coefficient is significant (p = 0.022).
5. Generates a **confusion matrix** using `pred_table()` and formats it as a labeled DataFrame.
6. Loads a **held-out test dataset** (`2.03.+Test+dataset.csv`, 19 observations).
7. Applies the trained model to the test data and evaluates predictions.

### Practice 1 — Simple Logistic Regression Exercise

**Notebook:** [`practice1/log_reg.ipynb`](practice1/log_reg.ipynb)

Applies simple logistic regression to the bank marketing dataset.

1. Loads `Bank_data.csv` (518 clients, 7 features).
2. Maps the target `y` from yes/no to 1/0.
3. Fits a Logit model using `duration` as the sole predictor.
4. Interprets the summary — Pseudo R² = 0.2121, duration coefficient is highly significant.

### Practice 2 — Binary Predictors Exercise

**Notebooks:** [`practice2/Binary+Predictors+in+a+Logistic+Regression+-+Exercise.ipynb`](practice2/Binary+Predictors+in+a+Logistic+Regression+-+Exercise.ipynb) (starter) / [`Binary+Predictors+in+a+Logistic+Regression+-+Exercise (1).ipynb`](<practice2/Binary+Predictors+in+a+Logistic+Regression+-+Exercise (1).ipynb>) (solved)

Mirrors Lab 2's approach on the bank dataset.

1. Loads `Bank_data.csv` and encodes the target.
2. Fits a Logit model with `duration` as predictor.
3. Visualizes the data with a scatter plot (duration vs. subscription).
4. Interprets the summary output.

### Practice 3 — Calculating Model Accuracy

**Notebook:** [`practice3/Calculating+the+Accuracy+of+the+Model+-+Exercise.ipynb`](practice3/Calculating+the+Accuracy+of+the+Model+-+Exercise.ipynb)

Extends the bank model with multiple predictors and introduces accuracy evaluation.

1. Fits a simple Logit model with `duration` only.
2. Visualizes the data with a scatter plot (duration vs. subscription).
3. Builds a **multivariate Logit model** with five predictors: `duration`, `interest_rate`, `march`, `credit`, `previous`.
4. Interprets the expanded summary output.
5. Implements a custom **`confusion_matrix()` function** that:
   - Predicts probabilities using the fitted model.
   - Bins predictions at a 0.5 threshold.
   - Computes the confusion matrix using `np.histogram2d`.
   - Calculates and returns the **accuracy** score.

### Practice 4 — Testing the Model

**Directory:** [`practice4/`](practice4/)

Provides the data for testing a trained model on unseen observations.

- `Bank_data.csv` — training data (518 rows).
- `Bank_data_testing.csv` — held-out test data (222 rows) with the same schema.
- The exercise notebook (preserved in `.ipynb_checkpoints/`) reuses the custom `confusion_matrix()` function from Practice 3 to evaluate model performance on the test set.

## Key Libraries

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `statsmodels` — Logit model fitting via Maximum Likelihood Estimation (MLE)

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels jupyterlab

# 3. Launch a notebook (e.g., Lab 1)
jupyter lab lab1/logistic_regression.ipynb
```

## Learning Objectives

- Understand the logistic (sigmoid) function and why linear regression fails for classification.
- Build and interpret a logistic regression model using `statsmodels.Logit`.
- Work with both continuous and binary categorical predictors.
- Read and interpret MLE-based regression summaries (coefficients, p-values, Pseudo R², Log-Likelihood).
- Construct and interpret a confusion matrix for binary classification.
- Implement a custom accuracy-evaluation function from scratch.
- Practice the train/test split workflow — training on one dataset and evaluating on a held-out test set.
