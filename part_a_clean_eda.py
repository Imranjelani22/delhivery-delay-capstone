
import os
import sys
import numpy as np
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES
)

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load original datasets
# --------------------------------------------------

raw = pd.read_csv(os.path.join(DATA_DIR, "raw_shipments.csv"))
hubs = pd.read_csv(os.path.join(DATA_DIR, "hubs.csv"))

# Remove exact duplicates for before-cleaning checks
dedup = raw.drop_duplicates().copy()

# --------------------------------------------------
# 3. Calculate before-cleaning checks
# --------------------------------------------------

raw_rows = len(raw)
rows_after_dedup = len(dedup)
duplicate_rows_removed = raw_rows - rows_after_dedup

missing_distance_before = dedup["distance_km"].isna().sum()

missing_weight_before = dedup["parcel_weight_kg"].isna().sum()

sentinel_weight_rows = (
    dedup["parcel_weight_kg"] > 100
).sum()

missing_declared_before = dedup["declared_value"].isna().sum()

missing_payment_before = dedup["payment_mode"].isna().sum()

service_spellings_before = (
    dedup["service_type"].nunique()
)

# --------------------------------------------------
# 4. Clean shipment data
# --------------------------------------------------

clean, medians = clean_shipments(raw)

service_spellings_after = (
    clean["service_type"].nunique()
)

# --------------------------------------------------
# 5. Build features and target
# --------------------------------------------------

modelling = build_features(clean, hubs)

rows_after_hub_merge = len(modelling)

modelling["is_delayed"] = (
    modelling["delivery_hours"] >
    modelling["promised_hours"]
).astype(int)

# --------------------------------------------------
# 6. Calculate data-quality and statistical checks
# --------------------------------------------------

delayed_count = int(modelling["is_delayed"].sum())

delay_rate = float(modelling["is_delayed"].mean())

null_columns = FEATURES + ["delivery_hours", "is_delayed"]

nulls_in_features_and_target = int(
    modelling[null_columns].isna().sum().sum()
)

hours = modelling["delivery_hours"]

hours_q1 = float(hours.quantile(0.25))
hours_q3 = float(hours.quantile(0.75))

hours_iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

hours_outlier_count = int(
    ((hours < hours_lower_bound) |
     (hours > hours_upper_bound)).sum()
)

declared_value_mean_inr = float(
    modelling["declared_value_inr"].mean()
)

declared_value_median_inr = float(
    modelling["declared_value_inr"].median()
)

log_declared_value_mean = float(
    modelling["log_declared_value"].mean()
)

log_declared_value_median = float(
    modelling["log_declared_value"].median()
)

# --------------------------------------------------
# 7. Create Part A checks CSV
# --------------------------------------------------

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_before),
    ("missing_payment_mode_before", missing_payment_before),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median)
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])

checks_df["value"] = pd.to_numeric(
    checks_df["value"], errors="coerce"
).round(4)

checks_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_checks.csv"),
    index=False
)

# --------------------------------------------------
# 8. Delay analysis by service, hub tier and distance
# --------------------------------------------------

modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True
)

group_results = []

grouping_dimensions = [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
    ("distance_band", "distance_band")
]

for column, dimension in grouping_dimensions:

    grouped = modelling.groupby(
        column,
        observed=False
    ).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    grouped = grouped.rename(columns={column: "group"})

    grouped["dimension"] = dimension

    grouped = grouped[
        ["dimension", "group", "shipments",
         "delayed", "delay_rate", "mean_hours"]
    ]

    group_results.append(grouped)

delay_by_group = pd.concat(
    group_results,
    ignore_index=True
)

delay_by_group["delay_rate"] = (
    delay_by_group["delay_rate"].round(4)
)

delay_by_group["mean_hours"] = (
    delay_by_group["mean_hours"].round(4)
)

delay_by_group.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_delay_by_group.csv"),
    index=False
)

# --------------------------------------------------
# 9. Numeric correlation matrix
# --------------------------------------------------

numeric_data = modelling.select_dtypes(
    include="number"
)

correlations = numeric_data.corr().reset_index()

correlations = correlations.rename(
    columns={"index": "column"}
)

correlations = correlations.round(4)

correlations.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_correlations.csv"),
    index=False
)

# --------------------------------------------------
# 10. Save complete modelling table
# --------------------------------------------------

numeric_columns = modelling.select_dtypes(
    include="number"
).columns

modelling[numeric_columns] = (
    modelling[numeric_columns].round(4)
)

modelling.drop(
    columns=["distance_band"],
    errors="ignore"
).to_csv(
    os.path.join(DATA_DIR, "modelling_table.csv"),
    index=False
)

# --------------------------------------------------
# 11. Print completion summary
# --------------------------------------------------

print("PART A EDA SCRIPT EXECUTED")
print("--------------------------------")
print("Raw rows:", raw_rows)
print("Rows after cleaning:", len(clean))
print("Rows after hub merge:", rows_after_hub_merge)
print("Delayed shipments:", delayed_count)
print("Delay rate:", round(delay_rate, 4))
print("Missing values in features and target:",
      nulls_in_features_and_target)

print("\nGenerated files:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")

