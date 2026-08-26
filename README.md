# California House Price Prediction using XGBoost

## Overview
A regression project predicting median house values for California districts using
the classic California Housing dataset (scikit-learn), built with XGBoost.

## Dataset
- Source: `sklearn.datasets.fetch_california_housing`
- 20,640 rows, 8 features (median income, house age, average rooms/bedrooms,
  population, average occupancy, latitude, longitude)
- Target: median house value (in $100,000s)
- No missing values

## Approach
1. **EDA** — correlation heatmap to inspect feature relationships
2. **Feature scaling** — StandardScaler applied before dimensionality analysis
3. **PCA** — used to examine explained variance across components and understand
   feature contribution
4. **Model** — XGBRegressor trained on an 80/20 train-test split
5. **Feature importance** — extracted and visualized from the trained XGBoost model
6. **Evaluation** — R² and MSE computed on both train and test sets
7. **Model persistence** — trained model saved with `pickle` for reuse

## Results
| Set   | R² Score | MSE   |
|-------|----------|-------|
| Train | 0.944    | 0.075 |
| Test  | 0.834    | 0.224 |

The gap between train and test R² suggests some overfitting — a natural next step
would be hyperparameter tuning (e.g., `GridSearchCV`/`RandomizedSearchCV`, already
imported) to improve generalization.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib

## Possible Next Steps
- Hyperparameter tuning to close the train/test performance gap
- Compare against Linear Regression / Random Forest as baselines
- Deploy via Streamlit for interactive predictions
