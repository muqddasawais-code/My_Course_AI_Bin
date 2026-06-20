import pandas as pd
# read the data from csv file
df = pd.read_csv('Csv-all-Datasets\\Startups in 2021 end.csv',delimiter="," ,parse_dates=[3],date_format={'Date Joined':'%d-%m-%Y'})  
print(df)

#query() to Select Data

#The query() method in Pandas allows you to select data using a more SQL-like syntax.

#select by valuation less than 50 and company name is FTX
df['Valuation ($B)'] = (
    df['Valuation ($B)']
    .astype(str)
    .str.replace('$', '', regex=False)
    .astype(float)
)

result1 = df.query('`Valuation ($B)` <= 50')
print(result1)


result1 = df[df['Valuation ($B)'] <= 50]
print(result1)

print("df- data types:",df.dtypes)

print("df infoo:",df.info())



#checking the no. of row and column using shape method
print("df shape:",df.shape)

# display the last three rows
print('Last three Rows:')
print(df.tail(5)) #by default is take 5 rows,

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()

#Summary of Statistics of DataFrame using describe() method.
print('Summary of Statistics of DataFrame using describe() method:')
print(df.describe())
print()

#selecting a column by using name of column
company= df['Company']
print(company)

#selecting multiple columns by using name of columns
company_industry=df[['Company','Industry']]
print(company_industry)


#Selecting a single row using .loc
second_row = df.loc[1]

#selecting a row by using .loc
row_2=df.loc[2]
print(row_2)

#selecting multiple rows by using .loc
second_row=df.loc[[1,4]]
print(second_row)
print()

#selecting a slice of rows by using .loc
slice_of_rows=df.loc[2:10:2] #start:stop:step
print(slice_of_rows)
print()

#conditional row selection by using .loc
condition_rows=df.loc[df['Industry']== 'E-commerce & direct-to-consumer']
print(condition_rows)
print()

#selecting a single column by using .loc
company_name=df.loc[:,'Company']
print(company_name)
print()

#selecting multiple columns by using .loc
company_industry=df.loc[1:5,['Company','Industry']]
print(company_industry)
print()

#selecting slicing of columns by using .loc
company_industry1=df.loc[2:10,'Company':'Industry'] #start:stop - here stop is included in the result   , START FROM COMPANY ENDS ON INDUSTRY
print(company_industry1)
print()

#selecting combined row and column by using .loc'
company_industry2=df.loc[df['Industry'] == 'E-commerce & direct-to-consumer', 'Country':'Industry'] 
print(company_industry2)    
print()


df_index_col=pd.read_csv('Csv-all-Datasets\\Startups in 2021 end.csv',delimiter="," ,parse_dates=[3],date_format={'Date Joined':'%d-%m-%Y'},index_col='Country')
print(df_index_col)
print("df index_col-data types:")
print(df_index_col.dtypes)
print("df index_col infoo:")
print(df_index_col.info())
print("df index_col shape:")
print(df_index_col.shape)
print("df index_col describe:")
print(df_index_col.describe())
#selecting a single row by using .loc w
print("df index_col head:")
print(df_index_col.head(4)) #by default it takes 5 rows, here we are taking 4 rows
print("df index_col tail:")
print(df_index_col.tail(4)) #by default it takes 5 rows, 


#selecting a single row by using .loc with index_col

second_row=df_index_col.loc['China']
print(second_row)
print()

#selecting multiple rows by using .loc with index_col
second_row1=df_index_col.loc[['China','India']]
print(second_row1)
print()

#selecting a slice of rows by using .loc with index_col
#second_row=df_index_col.loc['Date Joined':'Select Investors'] #start:stop - here stop is included in the result
#print(second_row)   

#conditional row selection by using .loc with index_col
second_row2=df_index_col.loc[df_index_col['Industry']== 'E-commerce & direct-to-consumer']
print(second_row2)
print()

#selecting a single column by using .loc with index_col
second_row3=df_index_col.loc[:,'Company']
print(second_row3)
print()

#selecting multiple columns by using .loc with index_col
second_row4=df_index_col.loc[:,['Company','Industry']]
print(second_row4)
print()

#selecting slicing of columns by using .loc with index_col
second_row5=df_index_col.loc[:,'Valuation ($B)':'Industry'] #start:stop - here stop is included in the result   , START FROM VALUATION ENDS ON INDUSTRY
print(second_row5)  
print()

#selecting combined row and column by using .loc with index_col
second_row6=df_index_col.loc[(df_index_col['Valuation ($B)'] >= '$100') , 'Company':'Industry']
print(second_row6)
print()

second_row7=df_index_col.loc[(df_index_col['Valuation ($B)'] < '$50') & (df_index_col['Company'] == 'Fanatics') ,'City':'Industry']
print(second_row7)
print()


#selecting a single row by using .iloc
second_row=df.iloc[1]
print(second_row)
print() 

#selecting multiple rows by using .iloc
second_row2=df.iloc[[1,5,9]]
print(second_row2)
print()

#selecting a slice of rows by using .iloc
second_row3=df.iloc[2:10:2] #start:stop:step
print(second_row3)
print()

#selecting a single column by using .iloc
second_row4=df.iloc[:,3] # here : means all rows and 3 means 4th column
print(second_row4)  
print()

#selecting multiple columns by using .iloc
second_row5=df.iloc[:,[0,5]] # here : means all rows and [0,5] means 1st and 6th column
print(second_row5)
print()

#selecting slicing of columns by using .iloc
second_row6=df.iloc[:,1:5]
print(second_row6) # here : means all rows and 1:5 means 2nd to 5th column - here 5 is not included in the result    
print()

#selecting combined row and column by using .iloc
second_row7=df.iloc[[1,6,5],3:5]
print(second_row7)
print()  #here 1:5 means 2nd to 5th row and 3:5 means 4th and 5th column - here 5 is not included in the result
#Case 3 : Using .iloc - ends here

# Next Run 
print("Next Run")



#Remove Rows/Columns from a Pandas DataFrame
# remove a row by index

df.drop(1,axis=0,inplace=True) # here 1 is index of row and axis=0 means we are dropping a row, inplace=True means we are dropping the row from original DataFrame
#delete row with index 1
df.drop(index=2,inplace=True)
#delete rows with index 3 and 5
df.drop([3,5],axis=0,inplace=True)
#display the modified DataFrame after deleting rows
print(df)


# remove a column by name
df.drop('City',axis=1,inplace=True)
# delete column with name 
df.drop(columns='Valuation ($B)',inplace=True)
#delete mutliple columns with name 
df.drop(['Company','Industry'],axis=1,inplace=True)
#DISPLAY THE MODIFIED DATAFRAME AFTER DELETING COLUMNS
print(df)

#creating filltered CSV file for sea born plot
df.to_csv(r'assignment 2/filtered_US_startup-2021_data.csv', index=False) 


