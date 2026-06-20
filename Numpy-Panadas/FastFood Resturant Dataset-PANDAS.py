import pandas as pd

df = pd.read_csv('Csv-all-Datasets\FastFoodRestaurants (1).csv', delimiter=',')
print(df)

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.columns)   
print(df['name'].unique())
print(df['name'].value_counts())
print(df[df['name'] == 'McDonald\'s'])
print(df[df['name'] == 'McDonald\'s']['city'].unique())
print(df[df['name'] == 'McDonald\'s']['city'].unique())
print(df[df['name'] == 'McDonald\'s']['latitude'].value_counts())
print(df[df['name'] == 'McDonald\'s']['latitude'].value_counts().head(10))



# display the last three rows
print('Last three Rows:')
print(df.tail(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()

#Summary of Statistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()



# access the Name column
name = df['name']
print("access the Name column: df : ")
print(name)
print()



# access multiple columns
name_province = df[['name','province']]
print("access multiple columns: df : ")
print(name_province)
print()


'''selection of rows using .loc'''

#Selecting a single row using .loc
second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

#Selecting a slice of rows using .loc
second_row3 = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()


#Conditional selection of rows using .loc
second_row4 = df.loc[df['name'] == 'subway']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:1,'latitude']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['name','province']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()


df_index_col = pd.read_csv('Csv-all-Datasets\FastFoodRestaurants (1).csv', delimiter=',')
print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

#Selecting a slice of rows using .loc
second_row3 = df_index_col.loc[1700:1740]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

#Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['name'] == "Arby's"]
print("#Conditional selection of rows using .loc")
print(second_row4)
print()


#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['name'] == "McDonald's",'postalCode':'websites']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

#using .iloc with index_col

#Selecting a single row using .iloc
second_row = df_index_col.iloc[0]
print("#Selecting a single row using .iloc")
print(second_row)
print()

#Selecting multiple rows using .iloc
second_row2 = df_index_col.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

#Selecting a slice of rows using .iloc
second_row3 = df_index_col.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

#Selecting a single column using .iloc
second_row5 = df_index_col.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df_index_col.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df_index_col.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df_index_col.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()


#Remove Rows/Columns from a Pandas DataFrame


# delete row with index 1
df.drop(1, axis=0, inplace=True)
# delete row with index 1
df.drop(index=2, inplace=True)
# delete rows with index 3 and 5
df.drop([3, 5], axis=0, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame - Remove Rows:")
print(df)

#Rename Labels in a DataFrame
# rename column '
df.rename(columns= {'province': 'province_nameChanged'}, inplace=True)

# delete column
df.drop('keys', axis=1, inplace=True)
# delete marital status column
df.drop(columns='latitude', inplace=True)
# delete height and profession columns
df.drop(['postalCode', 'websites'], axis=1, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame - Remove Columns:")
print(df)


#Example: Rename Row Labels
# rename column one index label
df.rename(index={0: 7}, inplace=True)
# rename columns multiple index labels
df.rename(mapper={1: 10, 2: 100}, axis=0, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)

#Sort Pandas DataFrame by Multiple Columns
# 1. Sort DataFrame 
df1 = df.sort_values(by=['country', 'name'])

print("Sorting by 'country' (ascending) and then by 'name' (ascending):\n")
print(df1.to_string(index=False))



#grouping Data in a Pandas DataFrame
grouped = df.groupby('address')['longitude'].sum()

print(grouped.to_string())
print("grouped :" , len(grouped))


# use dropna() to remove rows with any missing values
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)


# create a list named data
data = [2, 4, 6, 8]
# create Pandas array using data
array1 = pd.array(data)
print(array1)
"""<IntegerArray>
[2, 4, 6, 8]
Length: 4, dtype: Int64"""


# creating a pandas.array of integers
int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()

df.to_csv(r'assignment 2\filtered_FastFoodRestaurants_modified.csv', index=False)

