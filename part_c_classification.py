# Part C - Shipment Delay Classification
# Run from the project root:
#     python part_c_classification.py
#
# Required input:
#     data/modelling_table.csv
#
# Outputs:
#     outputs/part_c_models.csv
#     outputs/part_c_feature_scores.csv
#     outputs/part_c_selectk.csv
#     outputs/part_c_tuning.csv
#     outputs/part_c_thresholds.csv
#     outputs/part_c_final_test.csv
#     outputs/part_c_final_config.csv

import os
import ast
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# ---------------------------------------------------------------------
# 1. Load data and create the required stratified 70/15/15 split
# ---------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("Split sizes:")
print(f"Train: {len(X_train)} rows; delayed: {int(y_train.sum())}")
print(f"Validation: {len(X_val)} rows; delayed: {int(y_val.sum())}")
print(f"Test: {len(X_test)} rows; delayed: {int(y_test.sum())}")


# ---------------------------------------------------------------------
# Helper function: calculate classification metrics from predictions
# ---------------------------------------------------------------------
def classification_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def evaluate_classifier(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    return classification_metrics(y_eval, predictions)


# ---------------------------------------------------------------------
# 2. Blind baseline, Logistic Regression, Decision Tree, Random Forest
# ---------------------------------------------------------------------
model_rows = []

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_val)
baseline_metrics = classification_metrics(y_val, baseline_predictions)
model_rows.append({"model": "blind_baseline", **baseline_metrics})

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ]
)
logistic_model.fit(X_train, y_train)
logistic_metrics = evaluate_classifier(logistic_model, X_val, y_val)
model_rows.append({"model": "logistic_regression", **logistic_metrics})

# Decision Tree depth experiment. Ties are resolved in favour of
# the smaller depth because depths are checked in ascending order.
tree_depth_rows = []
best_tree_depth = None
best_tree_f1 = -1

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    tree_metrics = evaluate_classifier(tree, X_val, y_val)

    tree_depth_rows.append(
        {"max_depth": depth, **tree_metrics}
    )

    if tree_metrics["f1"] > best_tree_f1:
        best_tree_f1 = tree_metrics["f1"]
        best_tree_depth = depth

best_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=42,
)
best_tree.fit(X_train, y_train)
best_tree_metrics = evaluate_classifier(best_tree, X_val, y_val)
model_rows.append(
    {
        "model": f"decision_tree_depth_{best_tree_depth}",
        **best_tree_metrics,
    }
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
random_forest.fit(X_train, y_train)
forest_metrics = evaluate_classifier(random_forest, X_val, y_val)
model_rows.append({"model": "random_forest", **forest_metrics})

models_table = pd.DataFrame(model_rows)
models_table.to_csv(
    "outputs/part_c_models.csv",
    index=False,
    float_format="%.4f",
)

pd.DataFrame(tree_depth_rows).to_csv(
    "outputs/part_c_tree_depths.csv",
    index=False,
    float_format="%.4f",
)

print("\nModel comparison:")
print(models_table.to_string(index=False))


# ---------------------------------------------------------------------
# 3. Feature F-scores and Random Forest importance for all 17 features
# ---------------------------------------------------------------------
feature_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
feature_forest.fit(X_train, y_train)

selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": selector_all.scores_,
        "rf_importance": feature_forest.feature_importances_,
    }
)

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)
feature_scores["is_noise"] = feature_scores["feature"].isin(
    NOISE_COLUMNS
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
].sort_values("f_rank").reset_index(drop=True)

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
    float_format="%.4f",
)

noise_importance = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

print(
    "\nCombined Random Forest importance of noise columns:",
    round(float(noise_importance), 4),
)


# ---------------------------------------------------------------------
# 4. SelectKBest comparison for k = 5, 7, 10, 12, 17
# ---------------------------------------------------------------------
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    selectk_model.fit(X_train, y_train)
    val_predictions = selectk_model.predict(X_val)

    selectk_rows.append(
        {
            "k": k,
            "val_f1": f1_score(
                y_val, val_predictions, zero_division=0
            ),
        }
    )

selectk_table = pd.DataFrame(selectk_rows)
selectk_table.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
    float_format="%.4f",
)

print("\nSelectKBest comparison:")
print(selectk_table.to_string(index=False))


# ---------------------------------------------------------------------
# 5. Tuning on the seven selected features
# ---------------------------------------------------------------------
selector_7 = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
)
grid_search.fit(X_train_7, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)
random_search.fit(X_train_7, y_train)

grid_val_predictions = grid_search.best_estimator_.predict(X_val_7)
random_val_predictions = random_search.best_estimator_.predict(X_val_7)

default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
default_forest_7.fit(X_train_7, y_train)
default_val_predictions = default_forest_7.predict(X_val_7)

tuning_table = pd.DataFrame(
    [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": grid_search.best_score_,
            "val_f1": f1_score(
                y_val, grid_val_predictions, zero_division=0
            ),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(random_search.best_params_),
            "best_cv_accuracy": random_search.best_score_,
            "val_f1": f1_score(
                y_val, random_val_predictions, zero_division=0
            ),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": f1_score(
                y_val, default_val_predictions, zero_division=0
            ),
        },
    ]
)

tuning_table.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
    float_format="%.4f",
)

print("\nTuning comparison:")
print(tuning_table.to_string(index=False))


# ---------------------------------------------------------------------
# 6. Choose the winning configuration using validation F1
# Tie priority: default forest, grid search, randomized search.
# ---------------------------------------------------------------------
tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

winning_row = (
    tuning_table.assign(
        tie_order=tuning_table["search"].map(tie_priority)
    )
    .sort_values(
        ["val_f1", "tie_order"],
        ascending=[False, True],
    )
    .iloc[0]
)

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

# Required final pipeline: selector k=7 followed by Random Forest.
final_model = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **chosen_params,
            ),
        ),
    ]
)
final_model.fit(X_train, y_train)


# ---------------------------------------------------------------------
# 7. Select alert threshold from validation probabilities only
# Highest threshold with recall >= 0.80; if none, use 0.1.
# ---------------------------------------------------------------------
val_probabilities = final_model.predict_proba(X_val)[:, 1]
threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
]

threshold_rows = []

for threshold in threshold_values:
    val_predictions_at_threshold = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_rows.append(
        {
            "threshold": threshold,
            **classification_metrics(
                y_val, val_predictions_at_threshold
            ),
        }
    )

threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
    float_format="%.4f",
)

eligible_thresholds = threshold_table.loc[
    threshold_table["recall"] >= 0.80, "threshold"
]

if len(eligible_thresholds) > 0:
    chosen_threshold = float(eligible_thresholds.max())
else:
    chosen_threshold = 0.1

print("\nThreshold table:")
print(threshold_table.to_string(index=False))
print("\nChosen threshold:", chosen_threshold)


# ---------------------------------------------------------------------
# 8. Evaluate the test set exactly once, at the selected threshold
# Do not tune or change the model after viewing these test results.
# ---------------------------------------------------------------------
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)

final_test_metrics = classification_metrics(
    y_test, test_predictions
)

final_test_table = pd.DataFrame(
    [{"threshold": chosen_threshold, **final_test_metrics}]
)
final_test_table.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
    float_format="%.4f",
)

max_depth_value = chosen_params.get("max_depth")
if max_depth_value is None:
    max_depth_value = "unlimited"

final_config = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": max_depth_value,
            "min_samples_leaf": chosen_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ]
)

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False,
)

print("\nFinal configuration:")
print(final_config.to_string(index=False))

print("\nFinal test result:")
print(final_test_table.to_string(index=False))

print("\nPart C completed. All seven required CSV output files were saved in outputs/.")
