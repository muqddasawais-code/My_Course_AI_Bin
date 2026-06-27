import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

#________Load Dataset From CSV_______

df = pd.read_csv('Csv-all-Datasets/Space_Industry_Analytics_2010_2024 (1)csv.csv',delimiter=',')

print(df.head())


#____DATA Analysation Using Pandas________

print(df.head())
df.info()
print(df.shape)
print(df.describe())

#______NUmerical Analysis Using Numpy_______
revenue = df['Revenue_USD_M'].to_numpy()
launches = df['Launches'].to_numpy()
print(revenue)

print(np.mean(revenue))
print(np.median(revenue))
print(np.max(revenue))
print(np.min(revenue))
print(np.std(revenue))
print(np.var(revenue))

#_____Fill missing values____
df.ffill(df.mean(numeric_only=True),inplace=True) #it prevents from fake patterns 

launches = df['Launches'].to_numpy()
correlation = np.corrcoef(revenue, launches)
print(correlation)

performance = np.where(revenue > 1000,"High", "Low")
print(performance)

#______Data Visualization /Co-relation USING Sea-Born____

#_______Pairlot for Random Analysis Relationships Between Features________

numeric_cols = [
     "Launches","Successful","Failed","Revenue_USD_M",
    "Budget_Funding_USD_M","Employees","Rockets","Success_Rate_%"]

#_______Pair Plot__________

sns.pairplot(df[numeric_cols],diag_kind='kde')
plt.show() 
sns.pairplot(df,vars=[
        "Launches",
        "Budget_Funding_USD_M",
        "Employees",
        "Rockets",
        "Revenue_USD_M"])     #hue="Company",diag_kind="kde")#It may BREAK relation

plt.show()            ##working with numeric cols for a gernel overview now i find some  useful coloreation b/w cols

#_______Heatmap Co-relation Analysis__________
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#__________Sucess vs Failure Analysis__________ 
success = df["Successful"].sum()
failure = df["Failed"].sum()
plt.pie([success,failure],labels=["Success","Failure"],autopct='%1.2f%%')
plt.title("Mission Success Analysis")
plt.show()

#_________Company Revenue Growth Anaylsis_______
company_growth = df.groupby("Company")["Revenue_USD_M"].sum()
company_growth.plot(kind='bar')
plt.xlabel("Company")
plt.ylabel("Total Revenue (Million USD)")
plt.title("Company Revenue Growth")
plt.show()


#_________Revenue vs Badget Analysis__________
sns.scatterplot(data=df,x='Budget_Funding_USD_M',y='Revenue_USD_M')
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Budget vs Revenue")
plt.show()

#___________Sucess RATE BY Company Analysis___________
success_company = df.groupby( "Company")["Success_Rate_%"].mean() #return series
success_company.plot(kind='bar') #bivariate analysis 
plt.ylabel("Success Rate %")
plt.show()

#________Machine Learning (Regression for Revenue Prediction)______

#______Feature engineering + Encoding (All selected columns are numerical so encoding is not required)_____

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#_______Testing/Trainning______

y= df["Revenue_USD_M"]
X = df[["Launches","Employees","Budget_Funding_USD_M","Rockets"]]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#________Scaling Data________

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#________Model Training______

model = LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)

#_________Regression Metrics__________

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#_____Calculate metrics__________

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n____Regression Metrics____:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

#_____Actual vs Predicted Table_____

compare = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})

print("\n___Actual vs Predicted____:")
print(compare.head(20))

#______Plot Actual vs Predicted_______

plt.figure(figsize=(8,6))
plt.scatter(y_test,predictions)
plt.plot( [y_test.min(), y_test.max()],[y_test.min(), y_test.max()])
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted")
plt.show()
print(predictions)

#________Classification: Predict Success Category______________

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix, classification_report)

#______High Sucess=1,Low Success=0_________
df["Category"] = np.where(df["Success_Rate_%"] > 90,1,0)
print(df["Category"].value_counts())

# _____Features and target___
X = df[["Launches","Budget_Funding_USD_M","Employees","Rockets"]]
y = df["Category"]

# ____Split FIRST_____
X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)

# ____Train Model___
mol= RandomForestClassifier(n_estimators=100,random_state=42)
mol.fit(X_train, y_train)
pred = mol.predict(X_test)

#_______Confusion Matrics_________
accuracy = accuracy_score(y_test, pred)
print(" \n__Classification Accuracy:")
print("Accuracy:", round(accuracy,4))
print("\n _Confusion Matrix:")

print("\n__Classification Report:")
print(classification_report(y_test,pred))
cm = confusion_matrix( y_test,pred)
print(cm)

#______Visual Confusion Matrix________
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#________Clustring Feature Data______
from sklearn.cluster import KMeans
X = df[["Launches","Revenue_USD_M","Employees"]]

#______Scale Data____________
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train clustering Model
kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ___View Cluster Summary________
summary = df.groupby("Cluster")[["Launches","Revenue_USD_M","Employees"]].mean()

print(summary)

#___________Visualize Clusters________
plt.figure(figsize=(10,6))
plt.scatter( df["Revenue_USD_M"],df["Employees"],c=df["Cluster"])
plt.xlabel("Revenue")
plt.ylabel("Employees")
plt.title("Company Clusters")
plt.show()
