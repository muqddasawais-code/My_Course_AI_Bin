import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()   # turns interactive mode on
import seaborn as sns
 
#1.data preparing using np and pd
df=pd.read_csv('Csv-all-Datasets\placement.csv',delimiter=",")

print(df.head())

print(df.info())
print(df.shape)

df=df.iloc[:,1:]
print(df.head())

#2.EDA+ Feature selection
plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
plt.show()

#3.extract input & OUTPUT as independent /dependent variable
X=df.iloc[:,0:2]
Y=df.iloc[:,-1]
print(X)
print(Y)
print(X.shape)
print(Y.shape)

#5.test_train_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.1)
print(X_train,X_test,)


#4.scale data
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train = scalar.fit_transform(X_train)
print(X_train)
X_test= scalar.transform(X_test)
print(X_test)

#model training 
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train,Y_train.values, clf=clf, legend=2)
plt.show()








