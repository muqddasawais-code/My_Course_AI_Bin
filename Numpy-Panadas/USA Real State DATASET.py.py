import pandas as pd
df=pd.read_csv('Csv-all-Datasets\\RealEstate-USA.csv',delimiter=',')
print(df)
print()
print(df.dtypes)#data type
print(df.info())
print(df.describe())
print(df.size)
print(df.shape)


#select heads
print(df.head(4))
print()
#select tails
print(df.tail(7))
#select columns
print(df['price'])
print()

#selection query
second_row= df.query('city == \'Ponce\' or price > 100000')
print(second_row)
print(len(second_row))



# selection of multiple columns
print(df[['price','bath']])

#selection of single row/multiple row
singlerow=print(df.loc[2])
print(singlerow)
print()

#Remove Rows/Columns from a Pandas DataFrame
# remove a row by index
df.drop(1,axis=0,inplace=True) # here 1 is index of row and axis=0 means we are dropping a row, inplace=True means we are dropping the row from original DataFrame
#delete row with index 1
df.drop(index=2,inplace=True)
#delete rows with index 3 and 5
df.drop([3,5],axis=0,inplace=True)
#display the modified DataFrame after deleting rows
print(df)

#filtering data
filtered_data = df[df['city'] == 'Ponce']
print(filtered_data)

#filtering data with multiple conditions
filtered_data = df[(df['city'] == 'Ponce') & (df['bed'] > 3)]
print(filtered_data)

#filtering data with OR condition
filtered_data = df[(df['city'] == 'Ponce') | (df['bed'] > 3)]
print(filtered_data)    

#filtering data with NOT condition
filtered_data = df[~(df['city'] == 'Ponce')]
print(filtered_data)

#filtering data with isin() method.isin method is used to filter data based on multiple values in a column. It returns a boolean Series that indicates whether each element in the column is contained in the specified list of values.
filtered_data = df[df['city'].isin(['Ponce', 'San Juan'])]  
print(filtered_data)

# filtering data with between() method. The between() method is used to filter data based on a range of values in a column. It returns a boolean Series that indicates whether each element in the column is between the specified lower and upper bounds.
filtered_data = df[df['price'].between(67000 , 150000)]
print(filtered_data)

#filteing data by highest price and lowest price
highest_price = df['price'].max()
lowest_price = df['price'].min()
print("Highest Price:", highest_price)
print("Lowest Price:", lowest_price)


#remove a column by name
df.drop('bath',axis=1,inplace=True)

# delete column with name 
df.drop(columns='acre_lot',inplace=True)

#delete mutliple columns with name 
df.drop(['street','house_size'],axis=1,inplace=True)

#DISPLAY THE MODIFIED DATAFRAME AFTER DELETING COLUMNS
print(df)

df.to_csv(r'assignment 2\filtered USA Real Estate Data.csv', index=False)









