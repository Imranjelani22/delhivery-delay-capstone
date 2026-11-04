# Package the final Part C classifier for Part D.
# Run from the project root:
#     python package_model.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# ---------------------------------------------------------------------
# 1. Read the selected Part C configuration and historical modelling data
# ---------------------------------------------------------------------
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
df = pd.read_csv("data/modelling_table.csv")

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])

max_depth_value = config["max_depth"]
if pd.isna(max_depth_value) or str(max_depth_value).strip().lower() in {
    "none", "nan", "unlimited"
}:
    max_depth = None
else:
    max_depth = int(max_depth_value)

threshold = float(config["threshold"])


# ---------------------------------------------------------------------
# 2. Rebuild the exact stratified 70/15/15 split used in Part C
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 3. Rebuild and fit the selected final pipeline on training rows
# ---------------------------------------------------------------------
final_pipeline = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            ),
        ),
    ]
)

final_pipeline.fit(X_train, y_train)


# ---------------------------------------------------------------------
# 4. Save the model and its supporting configuration files
# ---------------------------------------------------------------------
model_path = "models/delay_model.pkl"
joblib.dump(final_pipeline, model_path)

pd.DataFrame({"threshold": [threshold]}).to_csv(
    "models/delay_threshold.csv",
    index=False,
)

pd.DataFrame({"feature": FEATURES}).to_csv(
    "models/feature_order.csv",
    index=False,
)

# Recompute the three historical cleaning medians using the shared
# Part A cleaning function, not a second copy of its cleaning logic.
raw_shipments = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw_shipments)

impute_values = pd.DataFrame(
    [
        {"column": column, "median": float(value)}
        for column, value in medians.items()
    ]
)

impute_values.to_csv(
    "models/impute_values.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------------------
# 5. Reload the saved model and compare test probabilities
# ---------------------------------------------------------------------
reloaded_pipeline = joblib.load(model_path)

memory_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_pipeline.predict_proba(X_test)[:, 1]

max_abs_probability_diff = float(
    np.max(np.abs(memory_probabilities - reloaded_probabilities))
)

reload_check = pd.DataFrame(
    [
        {"metric": "test_rows_scored", "value": int(len(X_test))},
        {
            "metric": "reload_max_abs_probability_diff",
            "value": max_abs_probability_diff,
        },
        {"metric": "threshold", "value": threshold},
    ]
)

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False,
    float_format="%.10f",
)


# ---------------------------------------------------------------------
# 6. Print completion summary
# ---------------------------------------------------------------------
print("Part D model packaging completed.")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows scored: {len(X_test)}")
print(f"Selected features (k): {selector_k}")
print(f"Random Forest estimators: {n_estimators}")
print(f"Random Forest max_depth: {max_depth}")
print(f"Random Forest min_samples_leaf: {min_samples_leaf}")
print(f"Alert threshold: {threshold}")
print(f"Reload max absolute probability difference: {max_abs_probability_diff:.12f}")

print("\nFiles saved:")
print(" - models/delay_model.pkl")
print(" - models/delay_threshold.csv")
print(" - models/feature_order.csv")
print(" - models/impute_values.csv")
print(" - outputs/part_d_reload_check.csv")

if len(X_test) != 180:
    raise ValueError(f"Expected 180 test rows, but found {len(X_test)}.")

if max_abs_probability_diff > 1e-9:
    raise ValueError(
        "Reloaded model probabilities do not match the in-memory model."
    )

print("\nReload check passed.")
