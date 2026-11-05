# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. Project Overview

This project develops a data-driven shipment delay intelligence pipeline using synthetic shipment data.

The objective is to analyse delivery performance, identify factors associated with shipment delays, build machine learning models, segment delivery hubs based on operational performance, and generate actionable recommendations for new shipments.

The project is developed as part of the IIT Roorkee AI/ML Capstone Project.

## 2. Repository Structure

```text
delhivery-delay-capstone/
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── src/
│   └── shipment_prep.py
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_correlations.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_b_checks.csv
│   ├── part_b_cv.csv
│   ├── part_b_gb_learning_rate.csv
│   ├── part_b_metrics.csv
│   ├── part_b_tree_depth.csv
│   ├── part_c_feature_scores.csv
│   ├── part_c_final_config.csv
│   ├── part_c_final_test.csv
│   ├── part_c_models.csv
│   ├── part_c_selectk.csv
│   ├── part_c_thresholds.csv
│   ├── part_c_tuning.csv
│   ├── hub_profile.csv
│   ├── part_d_separation.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   ├── tier_vs_segment_mismatch.csv
│   ├── part_d_reload_check.csv
│   └── new_shipment_actions.csv
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## 3. Part A – Data Preparation and Exploratory Data Analysis

### Data Cleaning

- Raw shipment records: 1,224.
- Exact duplicate rows removed: 24.
- Unique shipments after deduplication: 1,200.
- Missing distance values: 36.
- Missing parcel weight values: 24.
- Scanner sentinel weight records identified: 5.
- Missing declared value records: 12.
- Missing payment mode records: 18.
- Service type spellings standardized from 12 to 3.
- Median distance used for imputation: 314.95 km.
- Median parcel weight used for imputation: 1.83 kg.
- Median declared value used for imputation: INR 1,121.50.

### Delivery Delay Overview

- Total unique shipments: 1,200.
- Delayed shipments: 278.
- Overall delay rate: 23.17%.
- No missing values remained across the required features and target.

### Key Findings

Express shipments recorded the highest delay rate at 48.97%, followed by Standard at 20.56% and Economy at 8.57%.

Delay rates by hub tier:

- Tier 1: 4.00%.
- Tier 2: 21.50%.
- Tier 3: 44.00%.

The delay rate increased with shipment distance:

- 0–200 km: 2.19%.
- 201–400 km: 13.68%.
- 401–600 km: 43.64%.
- 601+ km: 85.23%.

## 4. Part B – Regression Modelling

The objective was to predict actual delivery time in hours.

### Dataset Split

- Training: 840 rows (70%).
- Validation: 180 rows (15%).
- Testing: 180 rows (15%).
- Random state: 42.

### Regression Models Evaluated

1. Linear Regression.
2. Decision Tree Regressor.
3. Random Forest Regressor.
4. Gradient Boosting Regressor.

### Validation Performance

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

Linear Regression was selected based on the lowest validation RMSE.

### Final Test Performance

- Selected model: Linear Regression.
- Test MAE: 3.5671 hours.
- Test MSE: 20.4624.
- Test RMSE: 4.5235 hours.
- Test R²: 0.9493.

Additional experiments included decision tree depth, Gradient Boosting learning rates, and five-fold cross-validation.

## 5. Part C – Shipment Delay Classification

The objective was to classify shipments as delayed or not delayed.

### Dataset Split

- Training: 840 rows.
- Validation: 180 rows.
- Testing: 180 rows.

### Classification Models Evaluated

1. Blind baseline.
2. Logistic Regression.
3. Decision Tree Classifier.
4. Random Forest Classifier.

Random Forest was selected for the final classification pipeline.

### Feature Selection and Hyperparameter Tuning

- SelectKBest feature selection was evaluated.
- Seven features were selected for the final configuration.
- GridSearchCV and RandomizedSearchCV were compared.
- Random Forest was configured with 100 estimators.
- Maximum depth: Unlimited.
- Minimum samples per leaf: 1.

### Threshold Selection

A probability threshold of 0.5 was selected for final classification.

The threshold was evaluated against alternative values to understand the trade-off between precision and recall.

### Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1-score | 72.09% |

Confusion matrix:

- True Negatives: 125.
- False Positives: 14.
- False Negatives: 10.
- True Positives: 31.

## 6. Part D – Hub Segmentation

The objective was to group delivery hubs based on their operational performance and identify hubs requiring management attention.

### Hub Profiling

- Total hubs analysed: 24.
- Clustering algorithms: K-Means.
- Number of clusters evaluated: 2, 3, 4 and 5.
- Final segmentation: Three operational segments.

### Final Hub Segments

| Segment | Hubs | Average Delivery Hours | Delay Rate |
|---|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% |
| Watch | 8 | 56.81 | 15.50% |
| Escalate | 9 | 64.65 | 46.00% |

### Segment Interpretation

**Standard**
- Relatively strong delivery performance.
- Low delay rate.
- Continue routine operational monitoring.

**Watch**
- Moderate delivery delays.
- Requires regular performance monitoring.
- Identify operational bottlenecks.

**Escalate**
- Highest average delivery time.
- Highest delay rate.
- Requires management attention and corrective action.

### Hub Tier versus Segment Analysis

The analysis also identified mismatches between existing hub tiers and their performance-based segments.

Four mismatches were identified for further operational review.

## 7. Model Packaging and Validation

The final Random Forest classification model was packaged for reuse.

### Saved Model Artefacts

- delay_model.pkl
- delay_threshold.csv
- feature_order.csv
- impute_values.csv

### Model Configuration

- Selected features: 7.
- Random Forest estimators: 100.
- Maximum depth: Unlimited.
- Minimum samples per leaf: 1.
- Alert threshold: 0.5.

### Model Reload Validation

The saved model was reloaded and tested against the same test data.

- Test rows scored: 180.
- Maximum absolute probability difference: 0.000000000000.
- Reload validation: Passed.

This confirms that the saved model reproduces the original prediction probabilities.

## 8. New Shipment Prediction and Operational Recommendations

The final prediction script was used to score 48 new shipments.

### Prediction Results

| Metric | Result |
|---|---:|
| New shipments scored | 48 |
| Alert threshold | 0.50 |
| Flagged shipments | 8 |
| Hub-manager escalations | 6 |
| Proactive customer calls | 2 |
| No action | 40 |

### Operational Action Rules

- Flagged shipment belonging to an Escalate segment: Escalate to hub manager.
- Other flagged shipments: Proactive customer call.
- Non-flagged shipments: No action.

The final output is saved as:

`outputs/new_shipment_actions.csv`

This output provides shipment-level predictions and recommended operational actions.

## 9. Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- GitHub
- Google Colab

## 10. Installation and Execution

### Clone the Repository

```bash
git clone https://github.com/Imranjelani22/delhivery-delay-capstone.git

cd delhivery-delay-capstone
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
python generate_data.py
```

### Run Part A – Data Preparation and EDA

```bash
python part_a_clean_eda.py
```

### Run Part B – Regression Modelling

```bash
python part_b_regression.py
```

### Run Part C – Classification

```bash
python part_c_classification.py
```

### Run Part D – Hub Segmentation

```bash
python part_d_segments.py
```

### Package and Validate the Model

```bash
python package_model.py
```

### Generate New Shipment Predictions

```bash
python predict_delay.py
```

**Execution note:** Run the scripts in the sequence above. The prediction script requires the saved model artefacts and the Part D hub segmentation outputs.

## 11. Project Status

- Part A – Data Preparation and Exploratory Data Analysis: Completed.
- Part B – Regression Modelling: Completed.
- Part C – Shipment Delay Classification: Completed.
- Part D – Hub Segmentation: Completed.
- Model Packaging and Reload Validation: Completed.
- New Shipment Prediction and Operational Recommendations: Completed.

## 12. Conclusion

The project demonstrates an end-to-end shipment delay intelligence pipeline covering data preparation, exploratory analysis, regression, classification, hub segmentation, model packaging, and shipment-level prediction.

The final pipeline provides both predictive insights and practical operational recommendations to help identify delayed shipments and improve delivery performance.

## Author

Imran Jilani

GitHub: https://github.com/Imranjelani22

IIT Roorkee AI/ML Capstone Project
