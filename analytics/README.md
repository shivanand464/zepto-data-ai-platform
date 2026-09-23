# Analytics Module

## Overview

This module performs exploratory data analysis, survival analysis, classification, and regression using the Titanic dataset provided by Seaborn.

The analysis includes:

* Dataset profiling
* Missing-value analysis and handling
* Survival analysis by passenger class and sex
* Data visualizations
* Logistic Regression classification
* Random Forest classification
* Classification evaluation using precision, recall, F1-score, and confusion matrices
* Linear Regression
* Regression evaluation using MAE, MSE, RMSE, and R²

---

## Dataset

The Titanic dataset is loaded using Seaborn:

```python
df = sns.load_dataset("titanic")
```

The dataset contains:

* **891 rows**
* **15 columns**

### Dataset Columns

The dataset contains the following columns:

* `survived`
* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`
* `class`
* `who`
* `adult_male`
* `deck`
* `embark_town`
* `alive`
* `alone`

---

## Data Profiling

The analysis examines:

* Dataset shape
* Column names
* First five rows
* Data types
* Missing values
* Numerical summary statistics
* Categorical summary statistics

The initial dataset contains missing values in:

| Column        | Missing Values |
| ------------- | -------------: |
| `age`         |            177 |
| `embarked`    |              2 |
| `deck`        |            688 |
| `embark_town` |              2 |

The remaining columns do not contain missing values.

---

## Missing-Value Handling

The following preprocessing steps were applied.

### 1. Remove `deck`

The `deck` column was removed because it contains a large proportion of missing values.

```python
df = df.drop(columns=["deck"])
```

### 2. Handle Missing `age`

Missing values in `age` were replaced using the median age.

```python
df["age"] = df["age"].fillna(df["age"].median())
```

### 3. Handle Missing `embarked`

Missing values in `embarked` were replaced using the most frequent value, or mode.

```python
df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)
```

### 4. Handle Missing `embark_town`

Missing values in `embark_town` were also replaced using the mode.

```python
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)
```

After preprocessing, the remaining dataset contains no missing values.

---

## Survival Analysis

The analysis examines survival rates overall and across passenger groups.

### Overall Survival Rate

The overall survival rate in the dataset was:

**38.38%**

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1               |        62.96% |
| 2               |        47.28% |
| 3               |        24.24% |

The analysis shows different survival rates across the three passenger classes.

---

## Survival by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        74.20% |
| Male   |        18.89% |

The analysis shows different survival rates between female and male passengers in this dataset.

---

## Survival by Passenger Class and Sex

| Passenger Class | Sex    | Survival Rate |
| --------------- | ------ | ------------: |
| 1               | Female |        96.81% |
| 1               | Male   |        36.89% |
| 2               | Female |        92.11% |
| 2               | Male   |        15.74% |
| 3               | Female |        50.00% |
| 3               | Male   |        13.54% |

This provides a more detailed view of survival rates by considering both passenger class and sex.

---

## Visualizations

The analysis generates three survival-analysis plots.

### Survival by Passenger Class

File:

```text
analytics/survival_by_class.png
```

This visualization shows survival rates across passenger classes.

### Survival by Sex

File:

```text
analytics/survival_by_sex.png
```

This visualization shows survival rates by sex.

### Survival by Passenger Class and Sex

File:

```text
analytics/survival_by_class_and_sex.png
```

This visualization shows survival rates by both passenger class and sex.

---

# Classification

## Objective

The classification models predict the `survived` variable.

The target variable is:

```text
survived
```

where:

* `0` represents did not survive
* `1` represents survived

---

## Classification Features

The following features were used:

* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`

The `alive` column was not used as a feature because it directly represents the survival outcome.

---

## Data Preprocessing

Categorical features were converted using one-hot encoding.

Categorical features:

* `sex`
* `embarked`

Numerical features:

* `pclass`
* `age`
* `sibsp`
* `parch`
* `fare`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

---

## Train-Test Split

The dataset was divided into:

* **Training samples:** 712
* **Testing samples:** 179

The test size was 20%.

A random state of `42` was used for reproducibility.

Stratification was used to maintain the class distribution between the training and testing datasets.

---

# Logistic Regression

A Logistic Regression model was trained to predict passenger survival.

The model used:

```python
LogisticRegression(max_iter=1000)
```

### Results

| Metric    | Result |
| --------- | -----: |
| Precision | 0.7931 |
| Recall    | 0.6667 |
| F1 Score  | 0.7244 |

### Confusion Matrix

```text
[[98 12]
 [23 46]]
```

The corresponding confusion-matrix visualization is saved as:

```text
analytics/logistic_confusion_matrix.png
```

---

# Random Forest

A Random Forest classification model was also trained.

The model used:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### Results

| Metric    | Result |
| --------- | -----: |
| Precision | 0.7869 |
| Recall    | 0.6957 |
| F1 Score  | 0.7385 |

### Confusion Matrix

```text
[[97 13]
 [21 48]]
```

The corresponding confusion-matrix visualization is saved as:

```text
analytics/random_forest_confusion_matrix.png
```

---

## Classification Evaluation Metrics

The classification models were evaluated using:

### Precision

Precision measures the proportion of predicted positive cases that were actually positive.

### Recall

Recall measures the proportion of actual positive cases that were correctly identified.

### F1 Score

F1 score combines precision and recall into a single metric.

### Confusion Matrix

The confusion matrix shows the number of:

* True negatives
* False positives
* False negatives
* True positives

---

# Regression

## Objective

The regression model predicts the continuous `fare` variable.

### Target

```text
fare
```

### Regression Features

The following features were used:

* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `survived`

The categorical `sex` feature was one-hot encoded.

The numerical features were passed through without encoding.

---

## Regression Model

A Linear Regression model was used:

```python
LinearRegression()
```

The regression dataset was divided into:

* **Training samples:** 712
* **Testing samples:** 179

A random state of `42` was used for reproducibility.

---

## Regression Results

| Metric |   Result |
| ------ | -------: |
| MAE    |  19.8104 |
| MSE    | 941.7679 |
| RMSE   |  30.6882 |
| R²     |   0.3914 |

### Mean Absolute Error

The MAE was:

```text
19.8104
```

### Mean Squared Error

The MSE was:

```text
941.7679
```

### Root Mean Squared Error

The RMSE was:

```text
30.6882
```

### R² Score

The R² score was:

```text
0.3914
```

---

## Regression Visualization

An actual-versus-predicted fare scatter plot is generated.

File:

```text
analytics/actual_vs_predicted_fare.png
```

The plot compares:

* Actual passenger fares
* Predicted passenger fares

---

# Generated Files

Running `analysis.py` generates the following visualization files:

```text
analytics/
│
├── analysis.py
│
├── README.md
│
├── survival_by_class.png
├── survival_by_sex.png
├── survival_by_class_and_sex.png
│
├── logistic_confusion_matrix.png
├── random_forest_confusion_matrix.png
│
└── actual_vs_predicted_fare.png
```

---

# How to Run

From the project root directory:

```powershell
python analytics\analysis.py
```

The script will:

1. Load the Titanic dataset.
2. Profile the dataset.
3. Display missing values and summary statistics.
4. Clean missing values.
5. Calculate survival rates.
6. Generate survival-analysis plots.
7. Prepare classification features.
8. Train Logistic Regression.
9. Train Random Forest.
10. Calculate classification metrics.
11. Generate confusion matrices.
12. Prepare regression features.
13. Train Linear Regression.
14. Calculate regression metrics.
15. Generate the actual-vs-predicted fare plot.

---

# Technologies Used

* Python 3
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn

---

# Reproducibility

The analysis uses fixed random states where applicable:

```text
random_state = 42
```

This helps produce reproducible train-test splits and Random Forest results.

---

# Summary

The Analytics module provides a complete analysis workflow covering:

* Exploratory data analysis
* Data-quality assessment
* Missing-value handling
* Survival analysis
* Data visualization
* Classification
* Classification evaluation
* Regression
* Regression evaluation

The module can be executed using:

```powershell
python analytics\analysis.py
```
