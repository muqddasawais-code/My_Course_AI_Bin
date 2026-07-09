"""
Telco Customer Churn - Logistic Regression Classification (FIXED)
====================================================================
Fixes applied vs original:
  1. Dropped TotalCharges from numeric features (it's ~ tenure * MonthlyCharges,
     near-perfectly collinear with the two, and was distorting their coefficients).
  2. Switched OneHotEncoder drop="if_binary" -> drop="first" so that every
     categorical feature (not just 2-category ones) drops a reference category.
     This removes the duplicated "No internet service" coefficient block that
     appeared 7 times in the original output.
"""

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

CSV_PATH = r'Csv-all-Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(CSV_PATH)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn"])
y = df["Churn"]

# FIX 1: TotalCharges removed (collinear with tenure * MonthlyCharges)
numeric_features = ["tenure", "MonthlyCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features and c != "TotalCharges"]
X = X.drop(columns=["TotalCharges"])

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Quick collinearity check for the write-up
corr_check = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
corr_check["Churn"] = corr_check["Churn"]
print("\nCorrelation matrix (tenure, MonthlyCharges, TotalCharges, Churn):")
print(corr_check.corr().round(3))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        # FIX 2: drop="first" instead of drop="if_binary"
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Logistic Regression (Fixed)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()  
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction (Fixed)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
plt.close()

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
coef_df.to_csv("coefficients_fixed.csv", index=False)

plt.figure(figsize=(9, 7))
top15 = coef_df.head(15).sort_values("coefficient")
colors = ["crimson" if c > 0 else "steelblue" for c in top15["coefficient"]]
plt.barh(top15["feature"], top15["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient (positive = increases churn risk)")
plt.title("Top 15 Logistic Regression Coefficients (Fixed)")
plt.tight_layout()
plt.show()
plt.close()
