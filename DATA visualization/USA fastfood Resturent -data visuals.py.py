import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
#load dataset from CSV (FAST FOOD RESTURANTS)
df=pd.read_csv('Csv-all-Datasets\FastFoodRestaurants (1).csv',delimiter=',')
print(df)
print(df.head)
print(df.tail)

#lineplot visualization
sns.set_theme(style='darkgrid')

sns.relplot(data=df,x='latitude',y='longitude',hue='name',palette='summer',kind='scatter')
plt.show()

sns.lineplot(data=df,x='latitude',y='address',hue='name')
plt.show()

sns.countplot(x='name',data=df)
plt.xticks(rotation=45)
plt.title('Count of Restaurants')
plt.show()

sns.barplot(x='province',y='name',data=df)
plt.show()

sns.sns.pairplot(data=df,height=1.5)
plt.show()

sns.catplot(x='country',y='province',data=df ,kind='bar')
plt.show()

sns.stripplot(x='country',y='province',data=df,dodge=True,palette='viridis',marker="*",size=10,alpha=0.2,jitter=0.5)
plt.show()


sns.swarmplot(x='country',y='name',data=df,size=10)
plt.show()

sns.histplot(df['latitude'], bins=10)
plt.title("Latitude Distribution")
plt.show()

sns.violinplot(df['name'],color='pink')
plt.show()


