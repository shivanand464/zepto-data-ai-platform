import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "analytics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 1. LOAD TITANIC DATASET ONCE + OFFLINE FALLBACK
# ============================================================

print("=" * 70)
print("1. LOADING TITANIC DATASET")
print("=" * 70)

try:
    df = sns.load_dataset("titanic")
    print("Loaded Titanic dataset using seaborn.")
except Exception:
    fallback_path = os.path.join(OUTPUT_DIR, "titanic.csv")

    if not os.path.exists(fallback_path):
        raise RuntimeError(
            "Could not download Titanic dataset and analytics/titanic.csv "
            "does not exist."
        )

    df = pd.read_csv(fallback_path)
    print("Loaded Titanic dataset from analytics/titanic.csv fallback.")

# Required offline fallback
df.to_csv(os.path.join(OUTPUT_DIR, "titanic.csv"), index=False)

print(f"Shape: {df.shape}")


# ============================================================
# 2. PROFILING
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET PROFILING")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe().to_string())

print("\nMissing-value percentages:")
missing_percent = (df.isnull().mean() * 100)
missing_percent = missing_percent[missing_percent > 0].sort_values(
    ascending=False
)
print(missing_percent.to_string())


# ============================================================
# 3. MISSING-VALUE HANDLING USING ASSIGNMENT THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING-VALUE HANDLING")
print("=" * 70)

# High missingness: drop deck.
# 5%-30%: impute age.
# Under 5%: drop affected rows.
#
# We preserve the measured percentages in the output.

print("\nMissing-value strategy:")

for column, pct in missing_percent.items():
    if pct < 5:
        strategy = "Drop affected rows"
    elif pct <= 30:
        strategy = "Impute"
    else:
        strategy = "Drop column / treat missing as unreliable"

    print(f"{column}: {pct:.2f}% -> {strategy}")

# deck has 77.22% missing, so it is dropped.
df = df.drop(columns=["deck"])

# age has 19.87% missing -> median imputation.
df["age"] = df["age"].fillna(df["age"].median())

# embarked and embark_town have <5% missing -> drop affected rows.
df = df.dropna(subset=["embarked", "embark_town"]).copy()

print("\nMissing values after cleaning:")
print(df.isnull().sum().to_string())


# ============================================================
# 4. SURVIVAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. SURVIVAL ANALYSIS")
print("=" * 70)

overall_survival = df["survived"].mean() * 100

sex_survival = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
)

class_survival = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
)

class_sex_survival = (
    df.groupby(["pclass", "sex"])["survived"]
    .mean()
    .mul(100)
)

print(f"\nOverall survival rate: {overall_survival:.2f}%")

print("\nSurvival by sex:")
print(sex_survival.to_string())

print("\nSurvival by passenger class:")
print(class_survival.to_string())

print("\nSurvival by class and sex:")
print(class_sex_survival.to_string())


# Boolean masking examples required by the assignment
female_survival = (
    df.loc[df["sex"] == "female", "survived"].mean() * 100
)

male_survival = (
    df.loc[df["sex"] == "male", "survived"].mean() * 100
)

first_class_survival = (
    df.loc[df["pclass"] == 1, "survived"].mean() * 100
)

third_class_survival = (
    df.loc[df["pclass"] == 3, "survived"].mean() * 100
)

female_first_class = (
    df.loc[
        (df["sex"] == "female") & (df["pclass"] == 1),
        "survived",
    ].mean() * 100
)

male_third_class = (
    df.loc[
        (df["sex"] == "male") & (df["pclass"] == 3),
        "survived",
    ].mean() * 100
)

print("\nBoolean-mask checks:")
print(f"Female survival: {female_survival:.2f}%")
print(f"Male survival: {male_survival:.2f}%")
print(f"1st-class survival: {first_class_survival:.2f}%")
print(f"3rd-class survival: {third_class_survival:.2f}%")
print(f"1st-class female survival: {female_first_class:.2f}%")
print(f"3rd-class male survival: {male_third_class:.2f}%")


# ============================================================
# 5. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. UNIVARIATE ANALYSIS")
print("=" * 70)


def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((series < lower) | (series > upper)).sum()

    return count, lower, upper


age_outliers, age_lower, age_upper = iqr_outlier_count(df["age"])
fare_outliers, fare_lower, fare_upper = iqr_outlier_count(df["fare"])

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]

print(f"\nAge IQR outliers: {age_outliers}")
print(f"Age lower bound: {age_lower:.4f}")
print(f"Age upper bound: {age_upper:.4f}")

print(f"\nFare IQR outliers: {fare_outliers}")
print(f"Fare lower bound: {fare_lower:.4f}")
print(f"Fare upper bound: {fare_upper:.4f}")

print("\nFare statistics:")
print(f"Mean:   {fare_mean:.4f}")
print(f"Median: {fare_median:.4f}")
print(f"Mode:   {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    fare_skew = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_skew = "left-skewed"
else:
    fare_skew = "not strictly determined by mean/median/mode ordering"

print(f"Fare distribution: {fare_skew}")


# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_histogram.png")
plt.close()

# Age boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_boxplot.png")
plt.close()

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fare_histogram.png")
plt.close()

# Fare boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fare_boxplot.png")
plt.close()


# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("6. CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

correlation_matrix = df[correlation_columns].corr()

print("\nRequired 6x6 correlation matrix:")
print(correlation_matrix.to_string())

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    square=True,
)
plt.title("Titanic Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()

# Find two strongest absolute off-diagonal correlations
pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        col1 = correlation_columns[i]
        col2 = correlation_columns[j]
        value = correlation_matrix.loc[col1, col2]

        pairs.append(
            {
                "feature_1": col1,
                "feature_2": col2,
                "correlation": value,
                "absolute_correlation": abs(value),
            }
        )

strongest_pairs = sorted(
    pairs,
    key=lambda x: x["absolute_correlation"],
    reverse=True,
)[:2]

print("\nTwo strongest absolute correlations:")

for pair in strongest_pairs:
    print(
        f"{pair['feature_1']} vs {pair['feature_2']}: "
        f"{pair['correlation']:.4f}"
    )


# ============================================================
# 7. MULTIVARIATE DATA STORY CHARTS
# ============================================================

print("\n" + "=" * 70)
print("7. MULTIVARIATE DATA STORY")
print("=" * 70)

# Chart 1: survival by class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/survival_by_class.png")
plt.close()

# Chart 2: survival by sex
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/survival_by_sex.png")
plt.close()

# Chart 3: survival by class and sex
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
plt.savefig(f"{OUTPUT_DIR}/survival_by_class_and_sex.png")
plt.close()

# Chart 4: fare by class and survival
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived",
)
plt.title("Fare Distribution by Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fare_by_class_and_survival.png")
plt.close()

# Chart 5: age, fare and survival
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.6,
)
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_vs_fare_survival.png")
plt.close()


# ============================================================
# 8. EXPLORATORY STANDARDIZATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("8. STANDARDIZATION CHECK")
print("=" * 70)

for column in ["age", "fare"]:
    mean_before = df[column].mean()
    std_before = df[column].std()

    standardized = (
        df[column] - mean_before
    ) / std_before

    print(f"\n{column}:")
    print(f"Before mean: {mean_before:.4f}")
    print(f"Before std:  {std_before:.4f}")
    print(f"After mean:  {standardized.mean():.6f}")
    print(f"After std:   {standardized.std():.6f}")


# ============================================================
# 9. CLASSIFICATION PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("9. CLASSIFICATION PREPARATION")
print("=" * 70)

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[classification_features]
y = df["survived"]

categorical_features = ["sex", "embarked"]

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nClass balance:")
print(y.value_counts().sort_index())
print(y.value_counts(normalize=True).sort_index())


# ============================================================
# 10. TRAINING-ONLY PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# 11. CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=5,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    ),
}

results = {}
predictions = {}
probabilities = {}


for model_name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    predictions[model_name] = y_pred
    probabilities[model_name] = y_prob

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_confusion_matrix.png"
    )
    plt.close()


# ============================================================
# 12. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for model_name in models:
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities[model_name],
    )

    auc_value = results[model_name]["auc"]

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_value:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png")
plt.close()


# ============================================================
# 13. DECISION TREE VISUALIZATION
# ============================================================

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=5,
            ),
        ),
    ]
)

decision_tree_pipeline.fit(X_train, y_train)

feature_names = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_model = (
    decision_tree_pipeline
    .named_steps["classifier"]
)

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8,
)

plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/decision_tree.png")
plt.close()


# ============================================================
# 14. IMBALANCE HANDLING COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("14. IMBALANCE HANDLING COMPARISON")
print("=" * 70)

imbalance_results = {}

# Baseline
baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_test)

imbalance_results["Baseline"] = {
    "precision": precision_score(y_test, baseline_pred),
    "recall": recall_score(y_test, baseline_pred),
    "f1": f1_score(y_test, baseline_pred),
}


# class_weight balanced
balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

balanced_pipeline.fit(X_train, y_train)
balanced_pred = balanced_pipeline.predict(X_test)

imbalance_results["Class Weight Balanced"] = {
    "precision": precision_score(y_test, balanced_pred),
    "recall": recall_score(y_test, balanced_pred),
    "f1": f1_score(y_test, balanced_pred),
}


# SMOTE only on training data
smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

smote_pipeline.fit(X_train, y_train)
smote_pred = smote_pipeline.predict(X_test)

imbalance_results["SMOTE"] = {
    "precision": precision_score(y_test, smote_pred),
    "recall": recall_score(y_test, smote_pred),
    "f1": f1_score(y_test, smote_pred),
}

imbalance_table = pd.DataFrame(imbalance_results).T

print(imbalance_table.to_string(float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 15. RANDOM FOREST GRID SEARCH + OOB SCORE
# ============================================================

print("\n" + "=" * 70)
print("15. RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                oob_score=True,
            ),
        ),
    ]
)

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_rf_model = best_rf.named_steps["classifier"]

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")
print(f"OOB score: {best_rf_model.oob_score_:.4f}")


# ============================================================
# 16. SAVE COMPLETE BEST CLASSIFICATION PIPELINE
# ============================================================

# Select the model with the highest F1 on the held-out test set.
best_model_name = max(
    results,
    key=lambda name: results[name]["f1"],
)

print(
    f"\nBest classifier by test F1: "
    f"{best_model_name}"
)

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            models[best_model_name],
        ),
    ]
)

best_pipeline.fit(X_train, y_train)

joblib_path = (
    f"{OUTPUT_DIR}/best_classification_pipeline.joblib"
)

joblib.dump(best_pipeline, joblib_path)

# Reload and verify raw-input prediction.
loaded_pipeline = joblib.load(joblib_path)
reload_predictions = loaded_pipeline.predict(
    X_test.head(5)
)

print(
    "Reloaded pipeline predictions:",
    reload_predictions.tolist(),
)


# ============================================================
# 17. REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("17. REGRESSION")
print("=" * 70)

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "survived",
]

X_reg = df[regression_features]
y_reg = df["fare"]

reg_categorical_features = ["sex"]

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived",
]

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            reg_numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            reg_categorical_features,
        ),
    ]
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
)

regression_model = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression()),
    ]
)

regression_model.fit(
    X_reg_train,
    y_reg_train,
)

regression_predictions = regression_model.predict(
    X_reg_test
)

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

n = len(y_reg_test)

p = (
    len(regression_model.named_steps["preprocessor"]
        .get_feature_names_out())
)

adjusted_r2 = (
    1
    - ((1 - r2) * (n - 1) / (n - p - 1))
)

residuals = (
    y_reg_test.to_numpy()
    - regression_predictions
)

print(f"MAE:         {mae:.4f}")
print(f"RMSE:        {rmse:.4f}")
print(f"R2:          {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")


# Actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(
    y_reg_test,
    regression_predictions,
    alpha=0.6,
)

min_value = min(
    y_reg_test.min(),
    regression_predictions.min(),
)

max_value = max(
    y_reg_test.max(),
    regression_predictions.max(),
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/actual_vs_predicted_fare.png"
)
plt.close()


# Residual plot
plt.figure(figsize=(8, 5))
plt.scatter(
    regression_predictions,
    residuals,
    alpha=0.6,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/regression_residuals.png"
)
plt.close()


# Simple heteroscedasticity interpretation:
# compare residual spread in lower vs upper predicted ranges.

prediction_series = pd.Series(
    regression_predictions
)

median_prediction = prediction_series.median()

low_spread = np.std(
    residuals[prediction_series <= median_prediction]
)

high_spread = np.std(
    residuals[prediction_series > median_prediction]
)

spread_ratio = (
    high_spread / low_spread
    if low_spread != 0
    else np.inf
)

if spread_ratio > 1.5 or spread_ratio < (1 / 1.5):
    heteroscedasticity = (
        "The residual spread differs substantially between "
        "lower and higher predicted fares, suggesting possible "
        "heteroscedasticity."
    )
else:
    heteroscedasticity = (
        "The residual spread is reasonably similar across the "
        "prediction range; there is no strong evidence of "
        "heteroscedasticity from this visual/spread check."
    )

print("\nHeteroscedasticity assessment:")
print(heteroscedasticity)


# ============================================================
# 18. MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("18. FINAL MODEL COMPARISON")
print("=" * 70)

classification_table = pd.DataFrame(results).T

print("\nClassification metrics:")
print(
    classification_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nRegression metrics:")
print(
    pd.DataFrame(
        {
            "MAE": [mae],
            "RMSE": [rmse],
            "R2": [r2],
            "Adjusted_R2": [adjusted_r2],
        },
        index=["Linear Regression"],
    ).to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 19. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYTICS MODULE COMPLETE")
print("=" * 70)

print("\nGenerated important files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.endswith((".png", ".csv", ".joblib")):
        print(f"- {OUTPUT_DIR}/{filename}")

print("\nBest classifier:")
print(best_model_name)

print("\nGridSearchCV best parameters:")
print(grid_search.best_params_)

print("\nGridSearchCV OOB score:")
print(f"{best_rf_model.oob_score_:.4f}")

print("\nRegression heteroscedasticity assessment:")
print(heteroscedasticity)