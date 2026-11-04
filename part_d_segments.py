# Part D - Hub Shipment Segmentation
# Run from the project root:
#     python part_d_segments.py
#
# Required inputs:
#     data/modelling_table.csv
#     data/hubs.csv
#
# Outputs:
#     outputs/hub_profile.csv
#     outputs/part_d_separation.csv
#     outputs/hub_segments.csv
#     outputs/segment_profile.csv
#     outputs/tier_vs_segment_mismatch.csv

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 1. Load the historical modelling table and hub reference data
# ---------------------------------------------------------------------
Path("outputs").mkdir(exist_ok=True)

shipments = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

required_shipment_columns = {
    "hub_id", "delivery_hours", "is_delayed", "distance_km", "n_stops"
}
required_hub_columns = {"hub_id", "hub_city", "hub_tier"}

missing_shipments = required_shipment_columns - set(shipments.columns)
missing_hubs = required_hub_columns - set(hubs.columns)

if missing_shipments:
    raise ValueError(
        f"modelling_table.csv is missing columns: {sorted(missing_shipments)}"
    )
if missing_hubs:
    raise ValueError(
        f"hubs.csv is missing columns: {sorted(missing_hubs)}"
    )


# ---------------------------------------------------------------------
# 2. Build one profile row per hub (fixed ascending hub_id order)
# ---------------------------------------------------------------------
hub_profile = (
    shipments.groupby("hub_id", as_index=False)
    .agg(
        shipments=("hub_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False,
    float_format="%.4f",
)

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

# Keep the hub order exactly as sorted above for both scaling and K-Means.
scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(hub_profile[profile_columns])


# ---------------------------------------------------------------------
# 3. Compare cluster separation for K = 2, 3, 4, 5
# Across-to-within ratio uses all 276 unordered pairs of 24 hubs.
# ---------------------------------------------------------------------
separation_rows = []
hub_pair_indices = list(combinations(range(len(hub_profile)), 2))

for k in [2, 3, 4, 5]:
    clustering = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = clustering.fit_predict(scaled_profiles)

    within_distances = []
    across_distances = []

    for i, j in hub_pair_indices:
        distance = float(
            np.linalg.norm(scaled_profiles[i] - scaled_profiles[j])
        )
        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    if not within_distances or not across_distances:
        raise ValueError(f"Cannot calculate separation ratio for K={k}.")

    ratio = float(np.mean(across_distances) / np.mean(within_distances))
    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True,
    )

    separation_rows.append(
        {
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": "-".join(str(size) for size in cluster_sizes),
        }
    )

separation_table = pd.DataFrame(separation_rows)
separation_table.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 4. Final business segmentation with K = 3
# Segment names are assigned by ascending mean hub delay rate.
# ---------------------------------------------------------------------
final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
hub_profile["cluster"] = final_kmeans.fit_predict(scaled_profiles)

cluster_delay_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

segment_names = ["Standard", "Watch", "Escalate"]
cluster_to_segment = {
    cluster_id: segment_names[position]
    for position, cluster_id in enumerate(cluster_delay_order)
}

hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_segment)

hub_segments = (
    hub_profile[["hub_id", "segment"]]
    .merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one",
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 5. Segment-level profile: means of the hub-level profile values
# ---------------------------------------------------------------------
segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "nunique"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

segment_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
segment_profile["_order"] = segment_profile["segment"].map(segment_order)
segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------------------
# 6. Compare observed segments with the playbook implied by hub tier
# Tier 1 -> Standard; Tier 2 -> Watch; Tier 3 -> Escalate.
# Accepts either numeric tiers (1, 2, 3) or strings such as "Tier 1".
# ---------------------------------------------------------------------
def implied_segment(tier):
    if pd.isna(tier):
        return None

    tier_text = str(tier).strip().lower()
    if tier_text in {"1", "tier 1", "tier1"}:
        return "Standard"
    if tier_text in {"2", "tier 2", "tier2"}:
        return "Watch"
    if tier_text in {"3", "tier 3", "tier3"}:
        return "Escalate"

    return None


hub_segments["_tier_expected_segment"] = (
    hub_segments["hub_tier"].apply(implied_segment)
)

mismatches = hub_segments.loc[
    hub_segments["_tier_expected_segment"].notna()
    & (
        hub_segments["segment"]
        != hub_segments["_tier_expected_segment"]
    ),
    ["hub_id", "segment", "hub_city", "hub_tier"],
].copy()

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 7. Print a concise completion summary
# ---------------------------------------------------------------------
print("Part D - Hub segmentation completed.")
print(f"Hub profile rows: {len(hub_profile)}")
print("\nCluster separation:")
print(separation_table.to_string(index=False))
print("\nSegment profile:")
print(segment_profile.to_string(index=False))
print("\nTier-versus-segment mismatches:")
print(mismatches.to_string(index=False))
print("\nFiles saved:")
print(" - outputs/hub_profile.csv")
print(" - outputs/part_d_separation.csv")
print(" - outputs/hub_segments.csv")
print(" - outputs/segment_profile.csv")
print(" - outputs/tier_vs_segment_mismatch.csv")
