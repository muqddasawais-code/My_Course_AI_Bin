import numpy as np

Serial_Number, List_Year, Assessed_Value, Sale_Amount, SalesRatio=np.genfromtxt('Csv-all-Datasets\RealEstate-USA.csv',delimiter=',',usecols=(0,1,5,6,7),dtype=float,unpack=True,skip_header=1,filling_values=0,invalid_raise=False)

print(Serial_Number)
print(List_Year)
print(Assessed_Value)
print(Sale_Amount)
print(SalesRatio)

# Statistics operations
print("Mean of Assessed_Value:", np.mean(Assessed_Value))
print("Median of Assessed_Value:", np.median(Assessed_Value))
print("Mean of Sale_Amount:", np.mean(Sale_Amount))
print("Median of Sale_Amount:", np.median(Sale_Amount))     
print("Mean of SalesRatio:", np.mean(SalesRatio))
print("Median of SalesRatio:", np.median(SalesRatio))
print('RealEstate_Sales 2001_2022  sales amount percentile_25 :',np.percentile(Sale_Amount,25))
print('RealEstate_Sales 2001_2022  sales amount percentile_50 :',np.percentile(Sale_Amount,50))
print('RealEstate_Sales 2001_2022  sales amount percentile_75 :',np.percentile(Sale_Amount,75))

# Correlation analysis
correlation_assessed_sale = np.corrcoef(Assessed_Value, Sale_Amount)[0, 1]
correlation_assessed_salesratio = np.corrcoef(Assessed_Value, SalesRatio)[0, 1]
correlation_sale_salesratio = np.corrcoef(Sale_Amount, SalesRatio)[0, 1]
print("Correlation between Assessed Value and Sale Amount:", correlation_assessed_sale)
print("Correlation between Assessed Value and Sales Ratio:", correlation_assessed_salesratio)       
print("Correlation between Sale Amount and Sales Ratio:", correlation_sale_salesratio)

# Calculate the difference between Assessed Value and Sale Amount
difference = Assessed_Value - Sale_Amount
print("Difference between Assessed Value and Sale Amount:", difference)
# Calculate the ratio of Assessed Value to Sale Amount
ratio = Assessed_Value / Sale_Amount
print("Ratio of Assessed Value to Sale Amount:", ratio)

#mathmeatical operations

print('RealEstate_Sales 2001_2022  sales amount square:',np.square(Sale_Amount))
print('RealEstate_Sales 2001_2022  sales amount power :',np.power(Sale_Amount,2))
print('RealEstate_Sales 2001_2022  sales amount sqrt :',np.sqrt(Sale_Amount))
print('RealEstate_Sales 2001_2022  sales amount abs :',np.abs(Sale_Amount))

#basic operations
add= Assessed_Value+Sale_Amount
subtraction=Assessed_Value-Sale_Amount
multiply=Assessed_Value*Sale_Amount
division=Assessed_Value/Sale_Amount

print('RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount Addition:',add)
print('RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount subtraction:',subtraction)    
print('RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount multiplication:',(multiply))
print('RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount division:',(division))


# Filter properties with Sale Amount greater than $500,000
high_value_properties = Sale_Amount > 500000
print("Properties with Sale Amount greater than $500,000:", high_value_properties)  

# Filter properties with Sales Ratio less than 0.8
low_sales_ratio_properties = SalesRatio < 0.8
print("Properties with Sales Ratio less than 0.8:", low_sales_ratio_properties)

# Filter properties listed in the year 2020
properties_2020 = List_Year == 2020
print("Properties listed in the year 2020:", properties_2020)

#trignometric operations
print('RealEstate_Sales 2001_2022  sales amount sine:',np.sin(Sale_Amount))
print('RealEstate_Sales 2001_2022  sales amount cosine:',np.cos(Sale_Amount))
print('RealEstate_Sales 2001_2022  sales amount tangent:',np.tan(Sale_Amount))      

#logical operations
print('RealEstate_Sales 2001_2022  sales amount greater than $500,000:',Sale_Amount > 500000)
print('RealEstate_Sales 2001_2022  sales amount less than $100,000:',Sale_Amount < 100000)
print('RealEstate_Sales 2001_2022  sales amount between $100,000 and $500,000:',(Sale_Amount > 100000) & (Sale_Amount < 500000))

#Trignometry-----
Sales_amountPie = (Sale_Amount/1000) +1

sine_values = np.sin(Sales_amountPie)
cosine_values = np.cos(Sales_amountPie)
tangent_values = np.tan(Sales_amountPie)

print("RealEstate_Sales 2001_2022  Sale_Amount- Sine values:", sine_values)
print("RealEstate_Sales 2001_2022  Sale_Amount Cosine values:", cosine_values)
print("RealEstate_Sales 2001_2022  Sale_Amount Tangent values:", tangent_values)
print("RealEstate_Sales 2001_2022  Sale_Amount - Exponential values:", np.exp(Sales_amountPie))

#Logarthmic value-----
log_array = np.log(Sales_amountPie)
log10_array = np.log10(Sales_amountPie)

print("RealEstate_Sales 2001_2022  Sale_Amount -- Natural logarithm values:", log_array)
print("RealEstate_Sales 2001_2022  Sale_Amount -- Base-10 logarithm values:", log10_array)

#Hyperbolic FUNCTIONS _______________

sinh_values = np.sinh(Sales_amountPie)
print("RealEstate_Sales 2001_2022  Sale_Amount   - Hyperbolic Sine values:", sinh_values)

cosh_values = np.cosh(Sales_amountPie)
print("RealEstate_Sales 2001_2022  Sale_Amount   - Hyperbolic Cosine values:", cosh_values)

tanh_values = np.tanh(Sales_amountPie)
print("RealEstate_Sales 2001_2022  Sale_Amount   -Hyperbolic Tangent values:", tanh_values)

asinh_values = np.arcsinh(Sales_amountPie)
print("RealEstate_Sales 2001_2022  Sale_Amount   -Inverse Hyperbolic Sine values:", asinh_values)

acosh_values = np.arccosh(Sales_amountPie)
print("RealEstate_Sales 2001_2022  Sale_Amount   -Inverse Hyperbolic Cosine values:", acosh_values)


#2Dimentional Array----

D2_Asses_Sales = np.array([Assessed_Value, Sale_Amount])

print ("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - " ,D2_Asses_Sales)

print("RealEstate_Sales 2001_2022 Assessed_value  Sale_AmountZameen.com Long Plus Lat - 2 dimentional arrary - dimension" , D2_Asses_Sales.ndim) 

print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - total number of elements" ,D2_Asses_Sales.size)

print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - gives size of array in each dimension" ,D2_Asses_Sales.shape)

print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - data type" ,D2_Asses_Sales.dtype) 


# Slicing array
D2_Asses_SalesSlice=  D2_Asses_Sales[:1,:4]
print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - Slicing array  " , D2_Asses_SalesSlice)
D2_Long_Lat_Slice2=  D2_Asses_Sales[:1, 5:15:4]
print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - Slicing array  " , D2_Asses_SalesSlice)



# Indexing array
D2_Asses_SalesSliceItemOnly=  D2_Asses_SalesSlice[0,2]
print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - Index array " , D2_Asses_SalesSliceItemOnly)
D2_Long_Lat_Slice2ItemOnly=  D2_Long_Lat_Slice2[0, 1]
print("RealEstate_Sales 2001_2022 Assessed_value  Sale_Amount - 2 dimentional arrary - index array  " , D2_Asses_SalesSliceItemOnly)

for elem in np.nditer(D2_Asses_Sales):
    print(elem)

for index, elem in np.ndenumerate(D2_Asses_Sales):
    print(index, elem)
D2 = np.reshape(D2_Asses_Sales, (1, 238))
print("RealEstate_Sales Assessed_value  Sale_Amount - 2 dimentional arrary - reshape : " , D2)
print("RealEstate_Sales Assessed_value  Sale_Amount - 2 dimentional arrary - reshape : Size " , D2.size)
print("RealEstate_Sales Assessed_value  Sale_Amount - 2 dimentional arrary - reshape : ndim " , D2.ndim)
print("RealEstate_Sales Assessed_value  Sale_Amount - 2 dimentional arrary - reshape : shape " , D2.shape)
print("RealEstate_Sales Assessed_value  Sale_Amount - 2 dimentional arrary - reshape : ndim " , D2.dtype)


