# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

This project develops a data-driven shipment delay intelligence pipeline using synthetic shipment data. The objective is to analyse delivery performance, identify factors associated with shipment delays, and build machine learning models to predict delivery time.

The project is developed as part of the IIT Roorkee AI/ML Capstone Project.

## Repository Structure

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
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_correlations.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_b_checks.csv
│   ├── part_b_cv.csv
│   ├── part_b_gb_learning_rate.csv
│   ├── part_b_metrics.csv
│   └── part_b_tree_depth.csv
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── requirements.txt
└── README.md
```

## Part A – Data Preparation and Exploratory Data Analysis

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

### Delay by Service Type

Express shipments recorded the highest delay rate at 48.97%, followed by Standard at 20.56% and Economy at 8.57%.

### Delay by Hub Tier

- Tier 1: 4.00% delay rate.
- Tier 2: 21.50% delay rate.
- Tier 3: 44.00% delay rate.

### Delay by Distance

The delay rate increased with distance:

- 0–200 km: 2.19%.
- 201–400 km: 13.68%.
- 401–600 km: 43.64%.
- 601+ km: 85.23%.

### Correlation Findings

The strongest correlations with delivery hours were:

- Promised hours: 0.8372.
- Distance: 0.4013.
- Number of stops: 0.3247.

### Declared Value Transformation

The mean declared value was INR 1,533.05, while the median was INR 1,121.50. After natural logarithm transformation, the mean and median were 7.0277 and 7.0224 respectively. The log transformation reduces the influence of highly skewed values.

## Part B – Regression Modelling

### Dataset Split

The modelling dataset was divided into three sets:

- Training: 840 rows (70%).
- Validation: 180 rows (15%).
- Testing: 180 rows (15%).

Random state: 42.

### Regression Models Evaluated

Four regression models were evaluated:

1. Linear Regression.
2. Decision Tree Regressor.
3. Random Forest Regressor.
4. Gradient Boosting Regressor.

### Validation Model Performance

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

### Decision Tree Depth Experiment

Maximum depths were tested to evaluate model complexity and overfitting.

- Best validation RMSE: 6.2796 hours.
- Best maximum depth: 6.
- Unlimited depth resulted in overfitting, with training RMSE of 0.0000 and validation RMSE of 7.8368.

### Gradient Boosting Learning Rate

Learning rates tested: 0.01, 0.1 and 1.0.

The best tested learning rate was 0.1, with validation RMSE of 4.9165 hours and validation R² of 0.9306.

### Cross-Validation

Five-fold cross-validation results:

- Linear Regression mean R²: 0.9261 (standard deviation 0.0048).
- Gradient Boosting mean R²: 0.9353 (standard deviation 0.0045).

### Final Model Selection

Linear Regression was selected based on the lowest validation RMSE among the four candidate models.

### Final Test Performance

- Selected model: Linear Regression.
- Test MAE: 3.5671 hours.
- Test MSE: 20.4624.
- Test RMSE: 4.5235 hours.
- Test R²: 0.9493.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- GitHub
- Google Colab

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/Imranjelani22/delhivery-delay-capstone.git
cd delhivery-delay-capstone
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic shipment data:

```bash
python generate_data.py
```

Run Part A:

```bash
python part_a_clean_eda.py
```

Run Part B:

```bash
python part_b_regression.py
```

## Project Status

- Part A – Data Preparation and Exploratory Data Analysis: Completed.
- Part B – Regression Modelling: Completed.
- Part C – Classification: Pending.
- Part D – Shipment Segmentation: Pending.
- Model Packaging and Prediction: Pending.

## Author

Imran Jilani

GitHub: https://github.com/Imranjelani22
