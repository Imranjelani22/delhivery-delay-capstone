
import pandas as pd
import numpy as np

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):

    clean_df = raw.copy()

    # 1. Remove exact duplicate rows
    clean_df = clean_df.drop_duplicates().copy()

    # 2. Standardise service type
    clean_df["service_type"] = (
        clean_df["service_type"].str.strip().str.title()
    )

    # 3. Fill missing payment mode
    clean_df["payment_mode"] = (
        clean_df["payment_mode"].fillna("Unknown")
    )

    # 4. Convert declared value into numeric INR
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch time into datetime (specifying custom format)
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"],
        format="%Y-%m-%d%H:%M"
    )

    # 6. Treat weight above 100 kg as missing
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # 7. Calculate or reuse medians
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            column: float(clean_df[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    for column in median_columns:
        clean_df[column] = clean_df[column].fillna(medians[column])

    return clean_df, medians


def build_features(clean, hubs):

    features_df = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left"
    )

    features_df["promised_hours"] = (
        features_df["service_type"].map(PROMISED_HOURS).astype(int)
    )

    features_df["is_cod"] = (
        features_df["payment_mode"].eq("COD").astype(int)
    )

    features_df["is_weekend"] = (
        features_df["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    features_df["dispatch_hour"] = (
        features_df["dispatch_time"].dt.hour.astype(int)
    )

    features_df["log_declared_value"] = np.log(
        features_df["declared_value_inr"]
    )

    return features_df
