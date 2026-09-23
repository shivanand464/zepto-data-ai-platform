import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# 1. LOAD DATASET

df = sns.load_dataset("titanic")

# 2. DATASET PROFILING

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print()
print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print()
print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string())

print()
print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print()
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print()
print("=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)
print(df.describe().to_string())

print()
print("=" * 60)
print("CATEGORICAL SUMMARY")
print("=" * 60)
print(df.describe(include="str").to_string())


# 3. MISSING VALUE HANDLING

print()
print("=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)

# The deck column contains a large number of missing values,
# so it is removed from the analysis.
df = df.drop(columns=["deck"])

# Fill missing age values with the median age.
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing categorical values with the most frequent value.
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

print("Missing values after cleaning:")
print(df.isnull().sum())


# 4. SURVIVAL ANALYSIS

print()
print("=" * 60)
print("OVERALL SURVIVAL RATE")
print("=" * 60)

survival_rate = df["survived"].mean() * 100

print(f"Overall survival rate: {survival_rate:.2f}%")


print()
print("=" * 60)
print("SURVIVAL BY PASSENGER CLASS")
print("=" * 60)

class_survival = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
)

print(class_survival)


print()
print("=" * 60)
print("SURVIVAL BY SEX")
print("=" * 60)

sex_survival = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
)

print(sex_survival)


print()
print("=" * 60)
print("SURVIVAL BY CLASS AND SEX")
print("=" * 60)

class_sex_survival = (
    df.groupby(["pclass", "sex"])["survived"]
    .mean()
    .mul(100)
)

print(class_sex_survival)

# 5. SURVIVAL VISUALIZATIONS

# Survival by passenger class
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_class.png")
plt.close()


# Survival by sex
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived",
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_sex.png")
plt.close()


# Survival by passenger class and sex
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex",
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_class_and_sex.png")
plt.close()


# 6. CLASSIFICATION MODEL PREPARATION

print()
print("=" * 60)
print("CLASSIFICATION MODEL PREPARATION")
print("=" * 60)

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[features]
y = df["survived"]


categorical_features = [
    "sex",
    "embarked",
]

numerical_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# 7. LOGISTIC REGRESSION

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)


logistic_model.fit(
    X_train,
    y_train,
)


logistic_predictions = logistic_model.predict(
    X_test
)


logistic_precision = precision_score(
    y_test,
    logistic_predictions,
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
)

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions,
)


print()
print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(
    f"Precision: {logistic_precision:.4f}"
)

print(
    f"Recall:    {logistic_recall:.4f}"
)

print(
    f"F1 Score:  {logistic_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(logistic_cm)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
    )
)

# 8. RANDOM FOREST

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
            ),
        ),
    ]
)


random_forest_model.fit(
    X_train,
    y_train,
)


random_forest_predictions = (
    random_forest_model.predict(X_test)
)


rf_precision = precision_score(
    y_test,
    random_forest_predictions,
)

rf_recall = recall_score(
    y_test,
    random_forest_predictions,
)

rf_f1 = f1_score(
    y_test,
    random_forest_predictions,
)

rf_cm = confusion_matrix(
    y_test,
    random_forest_predictions,
)


print()
print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(
    f"Precision: {rf_precision:.4f}"
)

print(
    f"Recall:    {rf_recall:.4f}"
)

print(
    f"F1 Score:  {rf_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(rf_cm)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        random_forest_predictions,
    )
)

# 9. CONFUSION MATRIX VISUALIZATIONS


# Logistic Regression confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "analytics/logistic_confusion_matrix.png"
)

plt.close()


# Random Forest confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "analytics/random_forest_confusion_matrix.png"
)

plt.close()


print()
print("=" * 60)
print("CLASSIFICATION ANALYSIS COMPLETE")
print("=" * 60)


# 10. REGRESSION MODEL PREPARATION

print()
print("=" * 60)
print("REGRESSION MODEL PREPARATION")
print("=" * 60)


# Features used to predict passenger fare.
regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "survived",
]


# Input features
X_reg = df[regression_features]

# Target variable
y_reg = df["fare"]


reg_categorical_features = [
    "sex",
]

reg_numerical_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived",
]


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            reg_categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            reg_numerical_features,
        ),
    ]
)


X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )
)


print(
    f"Training samples: {len(X_reg_train)}"
)

print(
    f"Testing samples: {len(X_reg_test)}"
)


# 11. LINEAR REGRESSION

regression_model = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor,
        ),
        (
            "regressor",
            LinearRegression(),
        ),
    ]
)


regression_model.fit(
    X_reg_train,
    y_reg_train,
)


regression_predictions = (
    regression_model.predict(X_reg_test)
)


# 12. REGRESSION METRICS

mae = mean_absolute_error(
    y_reg_test,
    regression_predictions,
)


mse = mean_squared_error(
    y_reg_test,
    regression_predictions,
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_reg_test,
    regression_predictions,
)


print()
print("=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print(
    f"MAE:  {mae:.4f}"
)

print(
    f"MSE:  {mse:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R2:   {r2:.4f}"
)


# 13. ACTUAL VS PREDICTED FARE PLOT

plt.figure(figsize=(8, 5))

plt.scatter(
    y_reg_test,
    regression_predictions,
    alpha=0.6,
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")

plt.tight_layout()

plt.savefig(
    "analytics/actual_vs_predicted_fare.png"
)

plt.close()

# 14. FINAL MESSAGE

print()
print("=" * 60)
print("REGRESSION ANALYSIS COMPLETE")
print("=" * 60)

print()
print("All Analytics tasks completed successfully.")
print("Generated plots:")
print("- analytics/survival_by_class.png")
print("- analytics/survival_by_sex.png")
print("- analytics/survival_by_class_and_sex.png")
print("- analytics/logistic_confusion_matrix.png")
print("- analytics/random_forest_confusion_matrix.png")
print("- analytics/actual_vs_predicted_fare.png")