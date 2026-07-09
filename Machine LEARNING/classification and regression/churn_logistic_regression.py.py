import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# 1. CONFIGURATION
CSV_PATH = r'Csv-all-Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 2. LOAD DATA

df = pd.read_csv(CSV_PATH)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# 3. CLEANING
# TotalCharges has some blank strings for customers with tenure = 0.
# Convert to numeric, coerce errors to NaN, then fill with 0 (they have no charges yet).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop the ID column, it has no predictive value
df = df.drop(columns=["customerID"])

# Target: convert Yes/No to 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Some columns use "No internet service" / "No phone service" as a
# separate category.  keep them as-is since they are informative,
# One-Hot-Encoding will handle them naturally.

# 4. FEATURES / TARGET SPLIT
X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# 5. TRAIN / TEST SPLIT (stratified to preserve churn ratio)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# 6. PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_features),
    ]
)

# 7. MODEL PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # dataset is imbalanced (~26% churn)
        random_state=RANDOM_STATE
    ))
])

# 8. HYPERPARAMETER TUNING (Grid Search for best C and penalty)
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["lbfgs"]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ------------------------------------------------------------------
# 9. EVALUATION ON TEST SET
# ------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print("TEST SET PERFORMANCE")
print("=" * 50)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# 10. CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.close()

# 11. ROC CURVE PLOT
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.close()

# 12. FEATURE IMPORTANCE (Top drivers of churn)

ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, cat_feature_names])

coefficients = best_model.named_steps["classifier"].coef_[0]
coef_df = pd.DataFrame({
    "feature": all_feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", key=abs, ascending=False)

print("\nTop 15 features driving churn prediction (by |coefficient|):")
print(coef_df.head(15).to_string(index=False))

plt.figure(figsize=(9, 7))
top15 = coef_df.head(15).sort_values("coefficient")
colors = ["crimson" if c > 0 else "steelblue" for c in top15["coefficient"]]
plt.barh(top15["feature"], top15["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient (positive = increases churn risk)")
plt.title("Top 15 Logistic Regression Coefficients")
plt.tight_layout()
plt.close()

