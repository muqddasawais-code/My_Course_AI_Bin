#Write a program to calculate the area of a rectangle (Length x Width)
lenght=3
width=2
print("Area of the rectangle is:", lenght*width)


#22. Take two numbers and print the result of the first raised to the power of the second (a^b).
a=5
b=2
print(a**b)


#Demonstrate the difference between / (division) and // (floor division) with the numbers 10 and 3
print(10/3)
print(10//3)


#Use the modulus operator % to find the remainder when 25 is devided by 4
print(25%4)


#Calculate the average of five numbers entered by the user
num1=float(input("enter first num"))
num2=float(input("enter second num"))
num3=float(input("enter third num"))
num4=float(input("enter fourth num"))
num5=float(input("enter fifth num"))
avge=(num1+num2+num3+num4+num5)/5
print("The average is:", avge)


#create a program that converts minutes into hours and remaining minutes.
min=int(input("enter minutes"))
hours=min/60
minutes=min%60
print("total hours:", hours)
print("remaining minutes:", minutes)



#Calculate the area of a circle where Area = \pi r^2 (Use 3.14 for \pi).
r=float(input("enter radius of circle"))
pi=3.14
area=pi*r**2
print("Area of the circle is:", area)   


#Find the cube of a number entered by the user.
NUM=int(input("enter a number"))
CUBE=NUM**3
print("The cube of the number is:", CUBE)


29. #Perform the calculation 10+5*2. Does Python follow PEMDAS? Prove it with code.
A=10+5*2
print(A)


#Write a program to calculate simple interest: (P \times R\times T)/100
p=float(input("enter principal amount"))
r=float(input("enter rate of interest"))
t=float(input("enter time in years"))
si=(p/r/t)/100
print("Simple Interest is:", si)


#Part4
#Compare two numbers entered by the user and print if the first is greater than the second.
a=int(input("enter first number"))
b=int(input("enter second number"))
print(a>b)


#Check if a user-entered number is even (Number % 2 == 0) and print the Boolean result.
num=int(input("enter a number"))
if num%2==0:
    print("The number is even.")
else:
    print("The number is odd.")



#Write a program that checks if a number is between 10 and 50 (inclusive) using and.
num=int(input("enter a number"))
if num>10 and num<=50:
    print("The number is between 10 and 50.")
else:
    print("The number is not between 10 and 50.")
    


#34.check if a string entered by the user is equal to "Python".
str=input("enter a string")
if str=="Python":
    print("string is equal to python")
else:
    print("string is not equal to python")


#Use the or operator to check if a user is either "Admin" or "Superuser".
user=input("enter user type")
if user=="admin" or user =="superstar":
    print("User is either Admin or Superuser.")
else:
    print("user is not admin or superstar")


#Demonstrate the not operator by reversing a Boolean variable.
a=True
print(not a)


#compare two floating-point numbers: 0.1+0.2-0.3. Explain the result
result=0.1+0.2-0.3
print(result)


#38. Take a user's age and check if they are NOT under 18.
age=int(input("enter your age"))
if not age>=18:
    print("You are under 18.")



#39.check if a number is positive and odd using logical operators.
num=int(input("enter a number"))
if num>0 and num%2!=0:
    print("The number is positive and odd.")
else:
    print("The number is not positive and odd.")



#compare the lengths of two strings provided by the user.
str1=input("enter first string")
print(len(str1))
str2=input("enter second string")
print(len(str2))
print(len(str1)>len(str2))


#Assignment Operators with Arithmetic

#41. Initialize a variable x = 10. Use += to add 5 to it.
x=10
x+=5
print(x) 


#Use - to subtract 3 from a variable price.
a=50
a-=3
print(a)


#43. Multiply a variable balance by 2 using the *= operator.
balance=20
balance*=2
print(balance)


#Divide a variable total by 4 using the / operator.
total=36
total/=4
print(total)


#Use ** to square a variable.
a=2
a**=2
print(a)


#Create a counter variable and increment it by 1 using assignment operators.
counter=0
counter+=1
print(counter)


#Use %= to find the remainder of a variable divided by 2 and update the variable.
num=15
num%=2
print(num)


#Use // to perform floor division on a variable and update it.
num=100
num//=3
print(num)

#49. Start with n = 2. In three lines of code using assignment operators, turn it into 20.
n=2
n=5
n=20
print(n)
#50. Create a variable with the value "drawer". Use indexing to print the first and last characters of the string.
str='drawer'
first_char=str[0]
last_char=str[-1]
print("First character:", first_char)
print("Last character:", last_char) 











