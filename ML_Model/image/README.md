# SpaceX Launch Analysis & Mission Outcome Prediction

This project performs exploratory data analysis (EDA) and builds machine learning models to predict **Mission Outcome** for SpaceX launches, using the `spaceX.2006.csv` dataset.

## Project Overview

The pipeline covers the full data science workflow:
1. Load and inspect the dataset
2. Clean and handle missing values
3. Visualize trends and relationships in the data
4. Engineer features from date/time fields
5. Encode categorical variables
6. Train and evaluate multiple classification models
7. Compare model performance and identify the best model

## Dataset

**File:** `Csv-all-Datasets/spaceX.2006.csv`

Key columns include:
- `Flight Number`, `Launch Date`, `Launch Time`, `Launch Site`
- `Vehicle Type`, `Payload Name`, `Payload Type`, `Payload Mass (kg)`, `Payload Orbit`
- `Customer Name`, `Customer Type`, `Customer Country`
- `Mission Outcome`, `Failure Reason`
- `Landing Type`, `Landing Outcome`

## Requirements

```
pandas
matplotlib
seaborn
numpy
scikit-learn
```

Install with:
```bash
pip install pandas matplotlib seaborn numpy scikit-learn
```

## Data Cleaning

Missing values are handled as follows:

| Column | Strategy |
|---|---|
| Payload Type, Payload Orbit, Customer Name, Customer Type, Customer Country | Filled with `"unknown"` |
| Failure Reason | Filled with `"No Failure"` |
| Landing Type, Landing Outcome | Filled with `"No Landing"` |
| Payload Mass (kg) | Filled with the column median |

Duplicate rows are also checked and reported.

## Exploratory Data Analysis & Visualizations

The script generates and saves the following charts (as PNG files, 300 DPI):

- Mission Outcome distribution (count plot)
- Number of launches per year
- Payload mass vs. launch year (scatter plot, colored by mission outcome)
- Vehicle type usage (bar chart)
- Payload type distribution (pie chart)
- Payload mass by mission outcome (box plot)
- Payload mass density (KDE plot)
- Payload mass by mission outcome (swarm plot)
- Payload mass by mission outcome (violin plot)
- Feature correlation heatmap
- Feature importance (from Random Forest)
- Model accuracy comparison (bar chart)

> **Note:** Output image paths in the script are currently hardcoded to a local Windows directory (`C:\Users\Awais\Documents\...`). Update these paths to a location on your own machine before running.

## Feature Engineering

- Extracted `Launch Year`, `Launch Month`, `Launch Day` from `Launch Date`
- Extracted `Launch Hour`, `Launch Minute` from `Launch Time`
- Created a binary `Heavy Payload` flag (1 if `Payload Mass (kg)` > 3000, else 0)
- Dropped original `Launch Date` and `Launch Time` columns
- Label-encoded all categorical columns

## Model Training

**Target variable:** `Mission Outcome`

**Features used:**
- Launch Site, Vehicle Type, Payload Type, Payload Mass (kg), Payload Orbit
- Customer Type, Customer Country
- Launch Year, Launch Month, Launch Hour, Heavy Payload

Data is split 80/20 (train/test) and scaled using `StandardScaler`.

### Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes

### Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Model Comparison

All model results are compiled into a comparison table and visualized in a bar chart (`model_accuracy_comparison.png`). The best-performing model (by accuracy) is automatically identified and printed at the end of the script.

## How to Run

1. Place `spaceX.2006.csv` inside a `Csv-all-Datasets` folder relative to the script (or update the path).
2. Update all hardcoded image-save paths to a valid directory on your system.
3. Run the script:
   ```bash
   python spacex_analysis.py
   ```
4. Review printed metrics in the console and generated plots in your chosen output folder.

## Possible Improvements
- Replace hardcoded file paths with relative paths or command-line arguments
- Add cross-validation for more robust model evaluation
- Perform hyperparameter tuning (e.g., GridSearchCV) for each model
- Handle class imbalance if `Mission Outcome` classes are skewed
- Save the trained best model (e.g., via `joblib`) for reuse
