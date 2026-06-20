customer_name = "Tim Kate"
customer_age = 59   
customer_height = 6.5
print("Customer Name:", customer_name, type(customer_name))
print("Customer Age:", customer_age, type(customer_age))
print("Customer Height:", customer_height, type(customer_height))

sum=customer_age+customer_height
print("Sum of age and height:", sum, type(sum))

sum=customer_age+5
customer_age=+5
print("Updated Customer Age:", customer_age, type(customer_age))

customer_info="tim kate is the primary customer of AbS GROUP.HE LIVES IN CALIFORNIA"
print(customer_info)
print(customer_info, type(customer_info))
print(len(customer_info))
for char in customer_info:
    print(char)

customer_list=["tim kate",54,6.5,"AbS",True]
print(customer_list)
print(type(customer_list))
print(customer_list[1])
print(customer_list[3])
print(type(customer_list[1]))
print(type(customer_list[3]))     
print(customer_list.append(3000),customer_list)
print(customer_list.insert(1,"lahore"),customer_list)
print(customer_list.remove(54),customer_list)
print(customer_list.pop(0),customer_list)
print(customer_list)
for x in customer_list:
 print(x)
 
 
customer_list[0]="changed value"
print(customer_list)

customer_tuple=("tim kate",54,6.5,"AbS",True)
print(len(customer_tuple))
print(customer_tuple)
print(customer_tuple[0])
print(customer_tuple[3])
print(type(customer_tuple[0]))
print(type(customer_tuple[3]))
for c in customer_tuple:
   print(c)

#customer_tuple[0]="chnaged value"
customer_set={"tim kate",54,6.5,"AbS", True}
print(customer_set)
print(type(customer_set))
print(len(customer_set))
for x in customer_set:
 print(x)

customer_set.add("lahore")
print(customer_set)

customer_set.discard(54)
print(customer_set)

customer_dic={"name":"tim kate", "age":54, "height":6.5,"company":"AbS", "status": True}
print(customer_dic)
print(type(customer_dic))
customer_dic["name"]="tim kate"
print(customer_dic["name"])
customer_dic["university"]="nicu"
print(customer_dic)
for x in customer_dic:
  print(x)
for x in customer_dic:
  print(customer_dic[x])

list=[1, 2, 2, 3, 2]
count = list.count(2)
print(f"The number 2 appears {count} times in the list.")