# Titanic Analytics Module

## 1. Overview

This module performs exploratory data analysis, statistical analysis, classification modeling, regression modeling, model evaluation, and model persistence using the Titanic dataset.

The module covers:

* Dataset profiling
* Missing-value analysis and handling
* Survival analysis
* Univariate analysis
* IQR-based outlier detection
* Fare distribution analysis
* Correlation analysis
* Multivariate data storytelling
* Exploratory standardization
* Classification using three models
* ROC/AUC evaluation
* Class-imbalance handling
* Random Forest hyperparameter tuning
* Regression modeling
* Regression residual analysis
* Model comparison
* Saving and reloading a complete classification pipeline

The main implementation is:

```text
analytics/
├── analysis.py
├── README.md
├── titanic.csv
├── *.png
└── best_classification_pipeline.joblib
```

---

## 2. Dataset

The module uses the Titanic dataset available through Seaborn.

The dataset contains **891 passenger records and 15 columns** before cleaning.

The dataset includes information such as:

* Passenger survival
* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Port of embarkation
* Passenger class labels
* Passenger type
* Adult male indicator
* Deck
* Embarkation town
* Survival label
* Whether the passenger travelled alone

### Dataset columns

| Column        | Description                                           |
| ------------- | ----------------------------------------------------- |
| `survived`    | Survival indicator: 0 = did not survive, 1 = survived |
| `pclass`      | Passenger class                                       |
| `sex`         | Passenger sex                                         |
| `age`         | Passenger age                                         |
| `sibsp`       | Number of siblings/spouses aboard                     |
| `parch`       | Number of parents/children aboard                     |
| `fare`        | Passenger fare                                        |
| `embarked`    | Port of embarkation                                   |
| `class`       | Passenger class as a categorical value                |
| `who`         | Passenger category                                    |
| `adult_male`  | Whether the passenger was an adult male               |
| `deck`        | Deck information                                      |
| `embark_town` | Embarkation town                                      |
| `alive`       | Survival label                                        |
| `alone`       | Whether the passenger travelled alone                 |

---

## 3. Dataset Profiling

The analysis begins by inspecting:

* Dataset shape
* Column names
* Data types
* First five rows
* Numerical summary statistics
* Missing-value counts
* Missing-value percentages

### Original dataset shape

```text
Rows:    891
Columns: 15
```

### Missing values

The original dataset contains missing values in:

| Column        | Missing Percentage |
| ------------- | -----------------: |
| `deck`        |             77.22% |
| `age`         |             19.87% |
| `embarked`    |              0.22% |
| `embark_town` |              0.22% |

---

## 4. Missing-Value Handling

The missing-value strategy follows percentage-based thresholds.

### Strategy

| Missing Percentage | Strategy           |
| ------------------ | ------------------ |
| Less than 5%       | Drop affected rows |
| 5%–30%             | Impute values      |
| More than 30%      | Drop the column    |

### Applied decisions

#### `deck`

`deck` has approximately 77.22% missing values.

Because this is above the high-missingness threshold, the column is removed.

```python
df = df.drop(columns=["deck"])
```

#### `age`

`age` has approximately 19.87% missing values.

Because this falls between 5% and 30%, missing values are replaced using the median.

```python
df["age"] = df["age"].fillna(df["age"].median())
```

#### `embarked` and `embark_town`

Both columns have approximately 0.22% missing values.

Because this is below 5%, rows with missing values in these columns are removed.

After cleaning, there are no remaining missing values in the analytical dataset.

---

## 5. Survival Analysis

The analysis examines survival at several levels.

### Overall Survival

The overall survival rate after cleaning is approximately:

```text
38.25%
```

### Survival by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        74.04% |
| Male   |        18.89% |

Female passengers have a substantially higher observed survival rate than male passengers in this dataset.

### Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1               |        62.62% |
| 2               |        47.28% |
| 3               |        24.24% |

The observed survival rate decreases from first class to third class.

### Survival by Class and Sex

| Class | Sex    | Survival Rate |
| ----: | ------ | ------------: |
|     1 | Female |        96.74% |
|     1 | Male   |        36.89% |
|     2 | Female |        92.11% |
|     2 | Male   |        15.74% |
|     3 | Female |        50.00% |
|     3 | Male   |        13.54% |

The combined class-and-sex analysis shows that both variables are associated with substantially different observed survival rates.

---

## 6. Boolean Masking Analysis

Boolean masking is used to directly calculate survival rates for selected passenger groups.

Examples include:

```python
df.loc[df["sex"] == "female", "survived"].mean()
```

and:

```python
df.loc[
    (df["sex"] == "female") & (df["pclass"] == 1),
    "survived"
].mean()
```

Selected results:

| Group                  | Survival Rate |
| ---------------------- | ------------: |
| Female passengers      |        74.04% |
| Male passengers        |        18.89% |
| First-class passengers |        62.62% |
| Third-class passengers |        24.24% |
| First-class females    |        96.74% |
| Third-class males      |        13.54% |

---

## 7. Univariate Analysis

### 7.1 Age Outliers

IQR-based outlier detection was applied to `age`.

Results:

```text
Age IQR outliers: 65
Lower bound: 2.50
Upper bound: 54.50
```

The IQR method identifies observations outside the range:

```text
Q1 - 1.5 × IQR
Q3 + 1.5 × IQR
```

These observations are reported as potential outliers rather than automatically removed.

### 7.2 Fare Outliers

IQR-based outlier detection was also applied to `fare`.

Results:

```text
Fare IQR outliers: 114
Lower bound: -26.7605
Upper bound: 65.6563
```

The negative lower bound is an artifact of the IQR calculation; fares themselves are non-negative.

### 7.3 Fare Statistics

The observed fare statistics after cleaning are:

| Statistic |   Value |
| --------- | ------: |
| Mean      | 32.0967 |
| Median    | 14.4542 |
| Mode      |  8.0500 |

The mean is considerably higher than the median, and the median is higher than the mode.

This indicates that the fare distribution is **right-skewed**, with a smaller number of high-fare observations increasing the mean.

### Univariate plots

The following plots are generated:

* `age_histogram.png`
* `age_boxplot.png`
* `fare_histogram.png`
* `fare_boxplot.png`

---

## 8. Correlation Analysis

The required six numerical columns are:

```text
survived
pclass
age
sibsp
parch
fare
```

The resulting 6×6 correlation matrix is:

|          | survived |  pclass |     age |   sibsp |   parch |    fare |
| -------- | -------: | ------: | ------: | ------: | ------: | ------: |
| survived |   1.0000 | -0.3355 | -0.0698 | -0.0340 |  0.0832 |  0.2553 |
| pclass   |  -0.3355 |  1.0000 | -0.3365 |  0.0817 |  0.0168 | -0.5482 |
| age      |  -0.0698 | -0.3365 |  1.0000 | -0.2325 | -0.1715 |  0.0937 |
| sibsp    |  -0.0340 |  0.0817 | -0.2325 |  1.0000 |  0.4145 |  0.1609 |
| parch    |   0.0832 |  0.0168 | -0.1715 |  0.4145 |  1.0000 |  0.2175 |
| fare     |   0.2553 | -0.5482 |  0.0937 |  0.1609 |  0.2175 |  1.0000 |

### Two strongest absolute correlations

The two strongest non-diagonal correlations are:

1. `pclass` vs `fare`: **-0.5482**
2. `sibsp` vs `parch`: **0.4145**

The negative correlation between passenger class and fare indicates that the numerical coding of passenger class is associated with fare levels: lower class numbers represent higher passenger classes, which generally have higher fares.

The positive correlation between `sibsp` and `parch` indicates that passengers travelling with more siblings/spouses also tend to have more parents/children aboard.

### Correlation visualization

Generated file:

```text
correlation_heatmap.png
```

---

## 9. Multivariate Data Story

Five multivariate charts are generated to examine relationships among multiple passenger characteristics.

### 9.1 Survival Rate by Passenger Class

Generated file:

```text
survival_by_class.png
```

**Interpretation:**

The survival rate varies substantially across passenger classes. First-class passengers have the highest observed survival rate, followed by second-class passengers, while third-class passengers have the lowest observed survival rate.

This indicates a strong association between passenger class and survival in the Titanic dataset.

---

### 9.2 Survival Rate by Sex

Generated file:

```text
survival_by_sex.png
```

**Interpretation:**

Female passengers have a substantially higher observed survival rate than male passengers.

The chart demonstrates a strong relationship between sex and survival outcomes in the dataset.

---

### 9.3 Survival Rate by Passenger Class and Sex

Generated file:

```text
survival_by_class_and_sex.png
```

**Interpretation:**

The combination of passenger class and sex provides more detail than either variable alone.

Female passengers have higher observed survival rates within each passenger class, while male passengers have lower survival rates. Differences between passenger classes are also visible within both sex groups.

---

### 9.4 Fare Distribution by Class and Survival

Generated file:

```text
fare_by_class_and_survival.png
```

**Interpretation:**

Fare distributions differ substantially between passenger classes. Higher passenger classes generally contain higher fares.

The chart also shows differences in fare distributions between passengers who survived and those who did not, although fare alone does not perfectly separate the two survival groups.

---

### 9.5 Age vs Fare by Survival

Generated file:

```text
age_vs_fare_survival.png
```

**Interpretation:**

The scatter plot shows the relationship between passenger age, fare and survival.

Passenger ages cover a broad range, while higher fares are concentrated among particular passenger groups and classes. The survival groups overlap considerably, indicating that age and fare individually do not completely separate survivors from non-survivors.

---

## 10. Exploratory Standardization Check

Standardization was explored for `age` and `fare`.

The standardization formula is:

```text
z = (x - mean) / standard deviation
```

### Age

Before standardization:

```text
Mean: 29.3152
Standard deviation: 12.9849
```

After standardization:

```text
Mean: approximately 0
Standard deviation: approximately 1
```

### Fare

Before standardization:

```text
Mean: 32.0967
Standard deviation: 49.6975
```

After standardization:

```text
Mean: approximately 0
Standard deviation: approximately 1
```

The actual machine-learning preprocessing also places scaling inside the model pipeline so that preprocessing is learned from the training data rather than from the full dataset.

---

# 11. Classification Modeling

## Objective

The classification task predicts whether a passenger survived.

Target:

```text
survived
```

Features:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

### Train/Test Split

The data is divided into:

```text
Training: 80%
Testing:  20%
```

Using:

```python
random_state=42
stratify=y
```

After the missing-value cleaning step:

```text
Training samples: 711
Testing samples: 178
```

---

## 12. Training-Only Preprocessing

The classification pipelines use a `ColumnTransformer`.

### Numerical features

The numerical features are:

```text
pclass
age
sibsp
parch
fare
```

Numerical preprocessing includes:

1. Median imputation
2. StandardScaler

### Categorical features

The categorical features are:

```text
sex
embarked
```

Categorical preprocessing includes:

1. Most-frequent imputation
2. One-hot encoding

All preprocessing is contained inside the machine-learning pipeline.

This ensures that preprocessing parameters are learned as part of model training rather than being calculated from the complete dataset before the train/test split.

---

# 13. Classification Models

Three classifiers are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The following metrics are reported:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Confusion matrix

---

## 14. Logistic Regression

Test-set results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.8090 |
| Precision | 0.7833 |
| Recall    | 0.6912 |
| F1        | 0.7344 |
| ROC-AUC   | 0.8610 |

Confusion matrix:

```text
[[97 13]
 [21 47]]
```

Generated file:

```text
logistic_regression_confusion_matrix.png
```

---

## 15. Decision Tree

Test-set results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7640 |
| Precision | 0.7600 |
| Recall    | 0.5588 |
| F1        | 0.6441 |
| ROC-AUC   | 0.8374 |

Confusion matrix:

```text
[[98 12]
 [30 38]]
```

Generated files:

```text
decision_tree_confusion_matrix.png
decision_tree.png
```

The decision tree visualization shows the learned tree structure and its first levels of decision rules.

---

## 16. Random Forest

Test-set results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.8090 |
| Precision | 0.7656 |
| Recall    | 0.7206 |
| F1        | 0.7424 |
| ROC-AUC   | 0.8196 |

Confusion matrix:

```text
[[95 15]
 [19 49]]
```

Generated file:

```text
random_forest_confusion_matrix.png
```

---

# 17. ROC Curves and AUC

ROC curves are generated for all three classifiers.

The test-set ROC-AUC values are:

| Model               | ROC-AUC |
| ------------------- | ------: |
| Logistic Regression |  0.8610 |
| Decision Tree       |  0.8374 |
| Random Forest       |  0.8196 |

Generated file:

```text
roc_curves.png
```

ROC-AUC summarizes how well the model separates the two target classes across classification thresholds.

---

# 18. Class Imbalance Analysis

The target distribution is:

| Survival | Count | Proportion |
| -------- | ----: | ---------: |
| 0        |   549 |     61.75% |
| 1        |   340 |     38.25% |

Because the classes are not perfectly balanced, three approaches are compared.

### Baseline

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.7833 |
| Recall    | 0.6912 |
| F1        | 0.7344 |

### Class Weight Balanced

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.7183 |
| Recall    | 0.7500 |
| F1        | 0.7338 |

### SMOTE

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.7353 |
| Recall    | 0.7353 |
| F1        | 0.7353 |

SMOTE is applied within an imbalanced-learn pipeline so that oversampling is performed on the training data rather than before the train/test split.

---

# 19. Random Forest Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is used to tune the Random Forest.

The parameter grid includes:

```text
n_estimators:
    100
    200

max_depth:
    None
    5
    10

max_features:
    sqrt
    log2
```

### Best Parameters

The selected configuration was:

```text
max_depth = 5
max_features = sqrt
n_estimators = 200
```

### Best Cross-Validation F1

```text
0.7408
```

### Random Forest OOB Score

```text
0.8214
```

The OOB score is calculated using:

```python
oob_score=True
```

---

# 20. Model Persistence

The best classification model based on test-set F1 was identified as:

```text
Random Forest
```

The complete preprocessing and classification pipeline is saved using `joblib`.

Generated file:

```text
best_classification_pipeline.joblib
```

The saved pipeline was then reloaded using `joblib.load()` and used to generate predictions on test data.

This verifies that the saved artifact can be restored and used without manually rebuilding the preprocessing steps.

---

# 21. Regression Analysis

## Objective

A Linear Regression model is used to predict passenger fare.

### Features

```text
pclass
sex
age
sibsp
parch
survived
```

Target:

```text
fare
```

The regression model uses the same training-only preprocessing principle:

* Numerical imputation
* Numerical standardization
* Categorical imputation
* One-hot encoding

---

## 22. Regression Results

The Linear Regression model produced:

| Metric      |  Result |
| ----------- | ------: |
| MAE         | 20.7146 |
| RMSE        | 42.4920 |
| R²          |  0.3232 |
| Adjusted R² |  0.2954 |

### Interpretation

The R² value indicates that the selected passenger characteristics explain part, but not all, of the variation in passenger fare.

The difference between R² and Adjusted R² accounts for the number of predictors included in the regression model.

---

# 23. Regression Visualizations

### Actual vs Predicted Fare

Generated file:

```text
actual_vs_predicted_fare.png
```

This plot compares observed fares against the fares predicted by the regression model.

Points closer to the diagonal reference line represent predictions closer to the observed fare.

### Residual Plot

Generated file:

```text
regression_residuals.png
```

The residual plot is used to examine whether residuals are randomly distributed around zero and whether their spread changes across predicted values.

---

# 24. Heteroscedasticity Analysis

The analysis compares residual spread between lower and higher predicted fare values.

The current result is:

```text
The residual spread differs substantially between lower and higher
predicted fares, suggesting possible heteroscedasticity.
```

This indicates that the variability of regression errors is not constant across the prediction range.

Therefore, the Linear Regression residuals show evidence of possible heteroscedasticity.

---

# 25. Final Classification Comparison

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |

The metrics show that model performance varies depending on the evaluation metric. Random Forest has the highest test F1 among the three models, while Logistic Regression has the highest ROC-AUC in this experiment.

The appropriate model choice should therefore consider the intended business objective and which evaluation metric matters most.

---

# 26. Generated Files

Running:

```powershell
python analytics\analysis.py
```

generates the following important artifacts.

### Dataset

```text
analytics/titanic.csv
```

### Exploratory analysis

```text
analytics/age_histogram.png
analytics/age_boxplot.png
analytics/fare_histogram.png
analytics/fare_boxplot.png
analytics/correlation_heatmap.png
```

### Survival analysis

```text
analytics/survival_by_class.png
analytics/survival_by_sex.png
analytics/survival_by_class_and_sex.png
```

### Multivariate analysis

```text
analytics/fare_by_class_and_survival.png
analytics/age_vs_fare_survival.png
```

### Classification

```text
analytics/logistic_regression_confusion_matrix.png
analytics/decision_tree_confusion_matrix.png
analytics/random_forest_confusion_matrix.png
analytics/roc_curves.png
analytics/decision_tree.png
```

### Regression

```text
analytics/actual_vs_predicted_fare.png
analytics/regression_residuals.png
```

### Saved model

```text
analytics/best_classification_pipeline.joblib
```

---

# 27. How to Run

From the project root:

```powershell
python analytics\analysis.py
```

The script automatically:

1. Loads the Titanic dataset using Seaborn.
2. Saves a local `titanic.csv` fallback.
3. Profiles the dataset.
4. Calculates missing-value percentages.
5. Applies the missing-value strategy.
6. Performs survival analysis.
7. Performs univariate analysis.
8. Calculates IQR outliers.
9. Calculates fare statistics and skewness.
10. Creates the correlation matrix.
11. Generates multivariate visualizations.
12. Checks standardization.
13. Splits the classification data.
14. Applies training-only preprocessing.
15. Trains Logistic Regression.
16. Trains Decision Tree.
17. Trains Random Forest.
18. Calculates classification metrics.
19. Generates ROC curves.
20. Compares imbalance-handling strategies.
21. Performs Random Forest GridSearchCV.
22. Reports the Random Forest OOB score.
23. Saves and reloads the classification pipeline.
24. Trains Linear Regression.
25. Calculates regression metrics.
26. Generates regression diagnostic plots.
27. Prints the final model comparison.

---

# 28. Technologies

The Analytics module uses:

* Python 3.12
* pandas
* NumPy
* Seaborn
* Matplotlib
* scikit-learn
* imbalanced-learn
* joblib

---

# 29. Reproducibility

The main machine-learning split uses:

```python
random_state=42
```

Classification uses an 80/20 train/test split with stratification.

Random Forest and SMOTE also use:

```python
random_state=42
```

This allows the analysis to be reproduced consistently under the same software and dataset conditions.

---

# 30. Key Findings

The exploratory analysis shows several clear patterns in the Titanic dataset:

1. Overall survival after cleaning is approximately 38%.
2. Female passengers have a substantially higher observed survival rate than male passengers.
3. First-class passengers have a substantially higher observed survival rate than third-class passengers.
4. Passenger class and sex show strong differences in survival outcomes.
5. Fare is right-skewed, with a mean substantially above the median.
6. Passenger class and fare have the strongest absolute correlation among the required numerical feature pairs.
7. Logistic Regression, Decision Tree and Random Forest produce different classification trade-offs.
8. Random Forest has the highest test-set F1 score among the three evaluated classifiers in this run.
9. Logistic Regression has the highest ROC-AUC among the three evaluated classifiers in this run.
10. The regression model explains part of the variation in fare, while the residual analysis suggests possible heteroscedasticity.

---

# 31. Module Status

The Analytics module includes:

* [x] Dataset profiling
* [x] Missing-value analysis
* [x] Threshold-based missing-value handling
* [x] Survival analysis
* [x] Boolean masking
* [x] IQR outlier detection
* [x] Fare mean, median and mode
* [x] Fare skewness analysis
* [x] Correlation matrix
* [x] Multivariate visualizations
* [x] Written chart interpretations
* [x] Exploratory standardization
* [x] Training-only preprocessing
* [x] Logistic Regression
* [x] Decision Tree
* [x] Random Forest
* [x] Accuracy
* [x] Precision
* [x] Recall
* [x] F1 score
* [x] Confusion matrices
* [x] ROC curves
* [x] ROC-AUC
* [x] Class-weight comparison
* [x] SMOTE comparison
* [x] Random Forest GridSearchCV
* [x] Random Forest OOB score
* [x] Linear Regression
* [x] MAE
* [x] RMSE
* [x] R²
* [x] Adjusted R²
* [x] Actual vs predicted plot
* [x] Residual plot
* [x] Heteroscedasticity assessment
* [x] Final model comparison
* [x] Joblib model persistence
* [x] Saved pipeline reload test

---

## 32. Summary

The Analytics module provides a complete Titanic dataset analysis workflow covering exploratory data analysis, statistical investigation, visualization, classification, class-imbalance handling, hyperparameter tuning, regression, diagnostics, and model persistence.

The workflow is reproducible through a single Python script:

```powershell
python analytics\analysis.py
```

The generated artifacts provide both the analytical results and the saved machine-learning pipeline required for further use.
