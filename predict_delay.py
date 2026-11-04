# Part D - Score the 48 new shipments and create operational actions.
# Run from the project root:
#     python predict_delay.py
#
# Uses the shared Part A functions in src/shipment_prep.py.
# It does not duplicate the cleaning or feature-engineering logic.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments


# ---------------------------------------------------------------------
# 1. Load input files and saved model artefacts
# ---------------------------------------------------------------------
Path("outputs").mkdir(exist_ok=True)

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold_df = pd.read_csv("models/delay_threshold.csv")
feature_order_df = pd.read_csv("models/feature_order.csv")
impute_values_df = pd.read_csv("models/impute_values.csv")

threshold = float(threshold_df.loc[0, "threshold"])

# Restore the exact historical medians used when the model was packaged.
medians = dict(
    zip(
        impute_values_df["column"],
        impute_values_df["median"].astype(float),
    )
)

required_median_keys = {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
}
if set(medians) != required_median_keys:
    raise ValueError(
        "impute_values.csv must contain exactly these columns: "
        f"{sorted(required_median_keys)}"
    )


# ---------------------------------------------------------------------
# 2. Clean new shipments using the shared Part A function and saved medians
# ---------------------------------------------------------------------
clean_new, _ = clean_shipments(new_shipments, medians=medians)

# Attach hub data and create model features using the shared Part A function.
featured_new = build_features(clean_new, hubs)

# Respect the saved feature order used when the model was trained.
feature_order = feature_order_df["feature"].tolist()

if feature_order != FEATURES:
    raise ValueError(
        "models/feature_order.csv does not match FEATURES in shipment_prep.py."
    )

missing_features = set(feature_order) - set(featured_new.columns)
if missing_features:
    raise ValueError(
        f"New shipment feature table is missing: {sorted(missing_features)}"
    )

X_new = featured_new.loc[:, feature_order]


# ---------------------------------------------------------------------
# 3. Predict delay probability and apply the alert threshold
# ---------------------------------------------------------------------
delay_probabilities = model.predict_proba(X_new)[:, 1]
delay_probabilities = np.round(delay_probabilities, 4)

flagged = (delay_probabilities >= threshold).astype(int)


# ---------------------------------------------------------------------
# 4. Attach hub segment and assign the operational action
# ---------------------------------------------------------------------
scored = pd.DataFrame(
    {
        "shipment_id": clean_new["shipment_id"].to_numpy(),
        "hub_id": clean_new["hub_id"].to_numpy(),
        "delay_probability": delay_probabilities,
        "flagged": flagged,
    }
)

scored = scored.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False,
)

if scored["segment"].isna().any():
    missing_hub_ids = (
        scored.loc[scored["segment"].isna(), "hub_id"].astype(str).unique().tolist()
    )
    raise ValueError(f"No segment found for hub IDs: {missing_hub_ids}")


def choose_action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"


scored["action"] = scored.apply(choose_action, axis=1)

# Required column order and original new_shipments.csv row order.
scored = scored[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]

# ---------------------------------------------------------------------
# 5. Validate and save the operational action list
# ---------------------------------------------------------------------
if len(scored) != 48:
    raise ValueError(f"Expected 48 new shipments, but found {len(scored)}.")

if scored["shipment_id"].isna().any() or scored["shipment_id"].duplicated().any():
    raise ValueError("shipment_id must be present and unique for all 48 shipments.")

if scored.isna().any().any():
    raise ValueError("The final action file contains missing values.")

expected_flagged = (
    scored["delay_probability"] >= threshold
).astype(int)

if not np.array_equal(scored["flagged"].to_numpy(), expected_flagged.to_numpy()):
    raise ValueError("Flagged values do not match the stored threshold rule.")

scored.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
    float_format="%.4f",
)

flagged_count = int(scored["flagged"].sum())
manager_escalations = int(
    (
        (scored["flagged"] == 1)
        & (scored["segment"] == "Escalate")
    ).sum()
)

print("Part D shipment scoring completed.")
print(f"New shipments scored: {len(scored)}")
print(f"Alert threshold: {threshold}")
print(f"Flagged shipments: {flagged_count}")
print(f"Hub-manager escalations: {manager_escalations}")
print("\nAction summary:")
print(scored["action"].value_counts().to_string())
print("\nFile saved: outputs/new_shipment_actions.csv")
print("\nFirst five scored shipments:")
print(scored.head().to_string(index=False))
