
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the required feature list from Part A
sys.path.insert(0, "src")
from shipment_prep import FEATURES

# --------------------------------------------------
# 1. Load the modelling table
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

# --------------------------------------------------
# 2. Train, validation and test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# --------------------------------------------------
# 3. Scaling checks: fit on training rows only
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": np.abs(X_train_scaled.mean(axis=0)).max()
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": np.abs(X_test_scaled.mean(axis=0)).max()
    }
])

os.makedirs("outputs", exist_ok=True)

checks.to_csv(
    "outputs/part_b_checks.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 4. Mean baseline
# --------------------------------------------------

mean_prediction = y_train.mean()
baseline_predictions = np.full(len(y_val), mean_prediction)

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }

baseline_metrics = calculate_metrics(y_val, baseline_predictions)

results = [{
    "model": "mean_baseline",
    "split": "val",
    **baseline_metrics
}]

# --------------------------------------------------
# 5. Decision Tree depth experiment
# --------------------------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)

    tree_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred))
    })

tree_depth_df = pd.DataFrame(tree_results)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
    float_format="%.4f"
)

# Select the depth with the lowest validation RMSE.
# If tied, prefer the smaller depth.
best_tree_row = tree_depth_df.sort_values(
    ["val_rmse"],
    ascending=True
).iloc[0]

best_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else int(best_tree_row["max_depth"])
)

# --------------------------------------------------
# 6. Four candidate regression models
# --------------------------------------------------

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    results.append({
        "model": name,
        "split": "val",
        **calculate_metrics(y_val, predictions)
    })

metrics_df = pd.DataFrame(results)

# --------------------------------------------------
# 7. Gradient Boosting learning-rate experiment
# --------------------------------------------------

learning_rate_results = []

for lr in [0.01, 0.1, 1.0]:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    predictions = gb_model.predict(X_val)

    learning_rate_results.append({
        "learning_rate": lr,
        "val_rmse": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "val_r2": r2_score(y_val, predictions)
    })

gb_lr_df = pd.DataFrame(learning_rate_results)

gb_lr_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 8. Cross-validation for the two best models
# --------------------------------------------------

candidate_metrics = metrics_df[
    metrics_df["model"] != "mean_baseline"
].copy()

top_two = candidate_metrics.sort_values(
    "RMSE"
).head(2)["model"].tolist()

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name in top_two:

    scores = cross_val_score(
        models[name],
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 9. Select final model using validation RMSE
# --------------------------------------------------

best_model_name = candidate_metrics.sort_values(
    "RMSE"
).iloc[0]["model"]

final_model = models[best_model_name]

# Score test data once
test_predictions = final_model.predict(X_test)
test_metrics = calculate_metrics(y_test, test_predictions)

test_row = pd.DataFrame([{
    "model": best_model_name,
    "split": "test",
    **test_metrics
}])

metrics_df = pd.concat(
    [metrics_df, test_row],
    ignore_index=True
)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 10. Print final results
# --------------------------------------------------

print("PART B REGRESSION SCRIPT COMPLETED")
print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

print("\nPart B checks:")
print(checks)

print("\nTree-depth results:")
print(tree_depth_df.round(4))

print("\nModel metrics:")
print(metrics_df.round(4))

print("\nLearning-rate results:")
print(gb_lr_df.round(4))

print("\nCross-validation results:")
print(cv_df.round(4))

print("\nFinal selected model:", best_model_name)
print("Final test R2:", round(test_metrics["R2"], 4))

print("\nAll five Part B output CSV files generated.")
