import numpy as np

latitude,longitude,postalCode=np.genfromtxt('Csv-all-Datasets/FastFoodRestaurants (1).csv',delimiter=",",usecols=(4,5,7),unpack=True,dtype=float,skip_header=1,filling_values=0,invalid_raise=False)
print(latitude)
print(longitude)
print(postalCode)

# fast-food Resturant---Statistical operations
print("FastFood longitude mean : " , np.mean(longitude))
print("FastFood longitude average: " , np.average(longitude))
print("FastFood longitude std: " , np.std(longitude))
print("FastFood longitude mod: " , np.median(longitude))
print("FastFood longitude percentile - 25: " , np.percentile(longitude,25))
print("FastFood longitude percentile  - 75: " , np.percentile(longitude,75))
print("FastFood longitude percentile  - 3: " , np.percentile(longitude,3))
print("FastFood longitude min : " , np.min(longitude))
print("FastFood longitude max : " , np.max(longitude))

# fast-food Resturant --Mathematical operations
print("FastFood postal code square: " , np.square(postalCode))
print("FastFood postal code sqrt: " , np.sqrt(postalCode))
print("FastFood postal code pow: " , np.power(postalCode,2))
print("FastFood postal code abs: " , np.abs(postalCode))           

# fast-food Resturant --Basic arithmetic operations

addition = longitude + latitude
subtraction = longitude - latitude
multiplication = longitude * latitude
division = longitude / latitude

print("FastFood longitude add: " , addition)
print("FastFood longitude subtract: " , subtraction)
print("FastFood longitude multiply: " , multiplication)
print("FastFood longitude divide: " , division)

# fast-food Resturant -- Trigonometric functions
postalCode_pie=(postalCode/1000)+1
print('Fast_Food_Restaurants postalCode  sin values:',np.sin(postalCode_pie))

print('Fast_Food_Restaurants postalCode  cos values:',np.cos(postalCode_pie))

print('Fast_Food_Restaurants postalCode  tan values:',np.tan(postalCode_pie))

print('Fast_Food_Restaurants postalCode  sin hyperbolic values:',np.sinh(postalCode_pie))

print('Fast_Food_Restaurants postalCode  cos inverse hyperbolic values:',np.acosh(postalCode_pie))

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(postalCode_pie)
log10_array = np.log10(postalCode_pie)   
print('Fast_Food_Restaurants postalCode  log values:',log_array)
print('Fast_Food_Restaurants postalCode  log10 values:',log10_array)

#fast-food Resturant -- Exponential functions
print('Fast_Food_Restaurants postalCode  exponential values:',np.exp(postalCode_pie))

#fast-food Resturant -- Rounding functions
print('Fast_Food_Restaurants postalCode  round values:',np.round(postalCode_pie))
print('Fast_Food_Restaurants postalCode  floor values:',np.floor(postalCode_pie))
print('Fast_Food_Restaurants postalCode  ceil values:',np.ceil(postalCode_pie)) 

#fast-food Resturant -- 2D array operations
# Create a 2D array using longitude and latitude
coordinates = np.array([longitude, latitude])
print("Fast_Food_Restaurants 2D array of coordinates:\n", coordinates)  

print('Fast_Food_Restaurants long/late 2 dimentional array :',coordinates.ndim)

print('Fast_Food_Restaurants long/late  2 dimentional array size :',coordinates.size)

print('Fast_Food_Restaurants long/late  2 dimentional array  shape:',coordinates.shape)

print('Fast_Food_Restaurants long/late  2 dimentional array  dtatype:',coordinates.dtype)

# Slicing the 2D array
D2longlatSlice=coordinates[1,:8]
print('Fast_Food_Restaurants long/late  2 dimentional array slicing:',D2longlatSlice)

#-----for loop------
for ele in np.nditer(coordinates):
    print(ele)

for index, ele in np.ndenumerate(coordinates):
    print(index,ele)


    #end of code
