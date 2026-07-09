
# Advertising Dataset - Linear Regression
# Libraries:
# Pandas, NumPy, Seaborn, Matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv(r"Csv-all-Datasets\advertising.csv",delimiter=",")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nInformation")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

# Missing Values

print("\nMissing Values")
print(df.isnull().sum())

# NumPy Analysis

tv = df["TV"].to_numpy()
sales = df["Sales"].to_numpy()

print("\nAverage TV Budget :", np.mean(tv))
print("Maximum TV Budget :", np.max(tv))
print("Minimum TV Budget :", np.min(tv))

print("Average Sales :", np.mean(sales))
print("Standard Deviation :", np.std(sales))

print("Correlation (TV vs Sales)")
print(np.corrcoef(tv, sales))

# Seaborn Visualizations

sns.set_style("whitegrid")

# Pairplot
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# TV vs Sales
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="TV", y="Sales")
plt.title("TV vs Sales")
plt.show()

# Radio vs Sales
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Radio", y="Sales")
plt.title("Radio vs Sales")
plt.show()

# Newspaper vs Sales
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Newspaper", y="Sales")
plt.title("Newspaper vs Sales")
plt.show()

# Prepare Data

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train Model

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation of Model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("Model Performance")
print("==============================")
print("MAE :", round(mae,3))
print("MSE :", round(mse,3))
print("RMSE:", round(rmse,3))
print("R2 Score:", round(r2,3))

# Coefficients

print("\nIntercept")
print(model.intercept_)

coef = pd.DataFrame({"Feature": X.columns,"Coefficient": model.coef_})

print("\nFeature Importance")
print(coef)

# Actual vs Predicted

result = pd.DataFrame({"Actual": y_test,"Predicted": y_pred})
print(result.head(10))

# Visualization

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()],[y.min(), y.max()],color="red")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Residual Plot

residuals = y_test - y_pred

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()