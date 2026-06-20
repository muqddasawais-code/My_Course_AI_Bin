import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('Csv-all-Datasets/titanic dataset.csv',delimiter=',')
print(df.head())
print(df.shape)
print(df.info)
print(df.size)
print(df.duplicated().sum())
print(df.describe)
print(df['Age'].max())
print(df['Age'].min())

#catagorial data analysis

sns.barplot(data=df,x='Pclass', y='Age',hue='Sex')
plt.show()

sns.displot(
    data=df[df['Survived']==0],
    x='Age',
    kde=True
)

plt.show()


sns.displot(
    data=df,
    x='Age',
    hue='Survived',
    kde=True
)

plt.show()

sns.countplot(x='Survived',data=df)
plt.xticks(rotation=45)
plt.title('Count of survived passanger')
plt.show()

sns.countplot(data=df,x='Embarked')
plt.show()

sns.countplot(x='Sex',data=df)
plt.xticks(rotation=45)
plt.title('Count of  passanger gender')
plt.show()

df['Survived'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()

#numerical data analysis
sns.histplot(data=df,x='Age',bins=20)
plt.show()

sns.displot(data=df,x='Age')
plt.show()

sns.boxplot(df['Age'])
plt.show()

sns.boxplot(data=df,x='Sex',y='Age',hue='Survived')
plt.show()


sns.boxplot(data=df,x='Fare')
plt.show()

sns.countplot(
    data=df,
    x='Survived',
    hue='Sex'
)

sns.violinplot(
    data=df,
    x='Survived',
    y='Age',
    hue='Sex'
)



