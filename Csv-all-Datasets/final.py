import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset from CSV

df = pd.read_csv('Csv-all-Datasets\Space_Industry_Analytics_2010_2024 (1)csv.csv',delimiter=',')

print(df.head())


# DATA PREPROCESSING


print(df.info())

print(df.isnull().sum())

# Fill missing values
df.fillna(method='ffill', inplace=True)

