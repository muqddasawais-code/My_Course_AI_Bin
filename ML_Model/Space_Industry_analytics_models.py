import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset from CSV

df = pd.read_csv('Csv-all-Datasets\Space_Industry_Analytics_2010_2024 (1)csv.csv',delimiter=',')

print(df.head())


# DATA analysation using Pandas
print(df.head())
print(df.info())
print(df.info())
print(df.shape)
print(df.describe)

#NUmerical analysis using numpy
revenue = df['Revenue_USD_M'].to_numpy()
launches = df['Launches'].to_numpy()

print(revenue)

print(np.mean(revenue))
print(np.median(revenue))
print(np.max(revenue))
print(np.min(revenue))
print(np.std(revenue))
print(np.var(revenue))

# Fill missing values if any
df.ffill(inplace=True)

launches = df['Launches'].to_numpy()

correlation = np.corrcoef(revenue, launches)

print(correlation)

performance = np.where(revenue > 1000,"High", "Low")
print(performance)

#Data visualization /co-relation USING Sea-born

#pairlot for random analysis Relationships between features
'''1-Correlations
2-Clusters
3-Trends
4-Distribution of each variable
5-Variables that may strongly influence (revenue)''' 

numeric_cols = [
    "Launches",
    "Successful",
    "Failed",
    "Revenue_USD_M",
    "Budget_Funding_USD_M",
    "Employees",
    "Rockets",
    "Success_Rate_%"
]

# Pair plot
sns.pairplot(
    df[numeric_cols],
    diag_kind='kde'
)

plt.show()  ##working with numeric cols for a gernel overview now i find some  useful coloreation b/w cols


#heatmap co-relation analysis
plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
     cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

#sucess vs failure analysis 

success = df["Successful"].sum()
failure = df["Failed"].sum()

plt.pie(
    [success,failure],
    labels=["Success","Failure"],
    autopct='%1.1f%%'
)

plt.title("Mission Success Analysis")
plt.show()

#company revenue growth anaylsis
company_growth = df.groupby("Company")["Revenue_USD_M"].sum()

company_growth.plot(kind='bar')
plt.xlabel("Company")
plt.ylabel("Total Revenue (Million USD)")
plt.title("Company Revenue Growth")
plt.show()


#revenue vs badget analysis
sns.scatterplot(data=df,x='Budget_Funding_USD_M',y='Revenue_USD_M')
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Budget vs Revenue")
plt.show()

#sucess RATE /company sucess rate Analysis
success_company = df.groupby( "Company")["Success_Rate_%"].mean() #return series
success_company.plot(kind='bar') #bivariate analysis 
plt.ylabel("Success Rate %")
plt.show()


#Machine Learning (regression for revenue prediction)

#feature engineering + Encoding (All selected columns are numerical so encoding is not required)

y= df["Revenue_USD_M"]
X = df[["Launches","Employees","Budget_Funding_USD_M","Rockets"]]

#test-tarin-split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
)

model = LinearRegression()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

print(predictions)
