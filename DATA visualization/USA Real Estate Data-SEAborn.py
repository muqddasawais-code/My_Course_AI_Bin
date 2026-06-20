import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Load the dataset from CSV file
df= pd.read_csv('assignment 2\\filtered USA Real Estate Data.csv', delimiter=',')

# Display the first few rows of the dataset
print(df.head( 100))

#display the summary statistics of the dataset
print(df.describe())

#display the data types of each column
print(df.dtypes)

#display the number of rows and columns in the dataset
print(df.shape)

#barplot visualization
sns.barplot(data=df,
            x='bed',
            y='price',
            hue='zip_code'
)
plt.show()




gap=df[['price', 'bed']]
print(gap)

#scatterplot visualization
sns.relplot(x='bed', y='price', data=gap, kind='scatter')
plt.title('Relationship between Price and Number of Beds')  
plt.xlabel('Number of Beds')
plt.ylabel('Price')     
plt.show()


#lineplot visualization
sns.relplot(x='bed', y='price', data=gap, kind='line',)
plt.title('Relationship between Price and Number of Beds')  
plt.xlabel('Number of Beds')
plt.ylabel('Price') 
plt.show()


#histpkot visualization
sns.displot(data=gap, x='price', kind='hist')
plt.show()

#kde plot visualization
sns.displot(data=gap ,x='price' , kind='kde')
plt.show()


#heatmap visualization 
temp = df.pivot_table(
    index='state',
    columns='bed',
    values='price',

)

sns.heatmap(temp, annot=True, cmap='viridis')
plt.show()
















