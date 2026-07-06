
from email.mime import image

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#--------------UNDERSTANDING DATASET----------

df=pd.read_csv('Csv-all-Datasets\spaceX.2006.csv',parse_dates=['Launch Date'])
df['Launch Time']=pd.to_datetime(df['Launch Time'],format='%H:%M')
print(df)

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)
print(df.dtypes)

#----------- CLEANING DATASET ----------

print("null values",df.isnull().sum())
print("duplicat values",df.duplicated().sum())

'''null values 
Flight Number         0
Launch Date           0   
Launch Time           0
Launch Site           0
Vehicle Type          0
Payload Name          0
Payload Type          3
Payload Mass (kg)     8
Payload Orbit         5
Customer Name         2
Customer Type         2
Customer Country      2
Mission Outcome       0
Failure Reason       33
Landing Type         20
Landing Outcome      20'''

#-----------FILLING MISSING VALUES------------

df['Payload Type']=df['Payload Type'].fillna('unknown')
df['Payload Orbit']=df['Payload Orbit'].fillna('unknown')
df['Customer Name']=df['Customer Name'].fillna('unknown')
df['Customer Type']=df['Customer Type'].fillna('unknown')
df['Customer Country']=df['Customer Country'].fillna('unknown')

df['Failure Reason']=df['Failure Reason'].fillna('No Failure')
df['Landing Type']=df['Landing Type'].fillna('No Landing')
df['Landing Outcome']=df['Landing Outcome'].fillna('No Landing')

df['Payload Mass (kg)']=df['Payload Mass (kg)'].fillna(df['Payload Mass (kg)'].median())

print(df.isnull().sum())
'''
Flight Number        0
Launch Date          0
Launch Time          0
Launch Site          0
Vehicle Type         0
Payload Name         0
Payload Type         0
Payload Mass (kg)    0
Payload Orbit        0
Customer Name        0
Customer Type        0
Customer Country     0
Mission Outcome      0
Failure Reason       0
Landing Type         0
Landing Outcome      0
dtype: int64'''


#----------Data visualization-------------

#----------Mission Outcome Distribution---------
  
plt.figure(figsize=(6,4))
sns.countplot(x='Mission Outcome',data=df)
plt.title('Mission Outcome Distribution')
plt.savefig( r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\mission_outcome_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#--------Launches Per Year----------
plt.figure(figsize=(10,5))
sns.countplot(
    x=df["Launch Date"].dt.year
)
plt.xticks(rotation=45)
plt.title("Number of Launches Per Year")
plt.xlabel("Year")
plt.ylabel("Launch Count")
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\launches_per_year.png",dpi=300,
            bbox_inches="tight")
plt.show()

#----------Payload Mass vs Launch Year------
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['Launch Date'].dt.year,
                y='Payload Mass (kg)',
                hue='Mission Outcome',
                data=df)
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\payload_mass_vs_launch_year.png",dpi=300,
            bbox_inches="tight")
plt.show()
#-----------Vehical Type and Usage-----------
vehicle=df['Vehicle Type'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(vehicle.index,vehicle.values)
plt.title('Vehicle distribution')
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\vehicle_distribution.png",dpi=300,
            bbox_inches="tight")
plt.show()
#------------Payload Type Distribution---------
payload = df["Payload Type"].value_counts()
plt.figure(figsize=(7,7))
plt.pie(
    payload.values,
    labels=payload.index,
    autopct="%1.1f%%"
)
plt.title("Payload Type Distribution")
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\payload_type_distribution.png",dpi=300,
            bbox_inches="tight")
plt.show() 

#----------Payload Mass by Mission Outcome---------
plt.figure(figsize=(8,5))
sns.boxplot(
    x="Mission Outcome",
    y="Payload Mass (kg)",
    data=df
)
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\payload_mass_by_outcome.png",dpi=300,
            bbox_inches="tight")
plt.show()


#--------------KDE PLOT------------
plt.figure(figsize=(8,5))
sns.kdeplot(
    data=df,
    x="Payload Mass (kg)",
    fill=True
)

plt.title("Payload Mass Density")
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\kde_plot.png",
            dpi=300,bbox_inches="tight")
plt.show()
#----------Swarm Plot ----------------
plt.figure(figsize=(8,5))

sns.swarmplot(
    x="Mission Outcome",
    y="Payload Mass (kg)",
    data=df
)
plt.title("Swarm Plot")
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\swarm_plot.png",
            dpi=300,bbox_inches="tight")
plt.show()

#-------------Payload Mass vs Mission Outcome----------
plt.figure(figsize=(8,5))

sns.violinplot(
    x="Mission Outcome",
    y="Payload Mass (kg)",
    data=df
)
plt.title("Payload Mass by Mission Outcome")
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\violin_plot.png",
            dpi=300,bbox_inches="tight")
plt.show()


#---------Machine Learning------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#-------------FEATURE ENGINEERING----------

#EXTRACT FEATURES FROM LAUNCH DATE

df['Launch Year']=df['Launch Date'].dt.year
df['Launch Month']=df['Launch Date'].dt.month
df['Launch Day']=df['Launch Date'].dt.day

# EXTRACT FEATURE FROM LAUNCH TIME

df['Launch Hour']=df['Launch Time'].dt.hour
df['Launch Minute']=df['Launch Time'].dt.minute

#CREATE HEAVY PAYLOAD FEATURE
import numpy as np
df['Heavy Payload']=np.where(df['Payload Mass (kg)']>3000,1,0)

# remove original date and time
df.drop(['Launch Date','Launch Time'],axis=1,inplace=True)
print(df.head())

#----------ENCODING--------------
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
categorial_columns=[
    'Flight Number',
    'Launch Site',
    'Vehicle Type',
    'Payload Name',
    'Payload Type',
    'Payload Orbit',
    'Customer Name',
    'Customer Type',
    'Customer Country',
    'Mission Outcome',
    'Failure Reason',
    'Landing Type',
    'Landing Outcome'
]

for col in categorial_columns:
    df[col]=le.fit_transform(df[col].astype(str))
print(df.head())
#------------Heatmap-----------
plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size":7}

)
plt.tight_layout()

plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\correlation_heatmap.png",dpi=300,
            bbox_inches="tight")
plt.show()
#---------- FEATURE SELECTION ----------
X = df[
    [
        "Launch Site",
        "Vehicle Type",
        "Payload Type",
        "Payload Mass (kg)",
        "Payload Orbit",
        "Customer Type",
        "Customer Country",
        "Launch Year",
        "Launch Month",
        "Launch Hour",
        "Heavy Payload"
    ]
]

y = df["Mission Outcome"]

#----------------traintest split-----------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
print('Training Data:',X_train.shape)
print('Testing Data :',X_test.shape)

#--------Standard Scaler---------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------LOGISTIC REGRESSION model-------------

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
lr_predict=lr.predict(X_test)
print('----Logistic regression----')
#-----------Evalution------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
print('accuracy:',accuracy_score(y_test,lr_predict))
print('precision:',precision_score(y_test,lr_predict))
print('Recall:',recall_score(y_test,lr_predict))
print('F1 score:',f1_score(y_test,lr_predict))
print('Confusion matrix:',confusion_matrix(y_test,lr_predict))
print('Classification report:',classification_report(y_test,lr_predict))

#----------DECISION TREE----------
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
dt_predict= dt.predict(X_test)

print('-----Decision Tree-----')
print('Accuracy:',accuracy_score(y_test ,dt_predict))
print('Precision:',precision_score(y_test ,dt_predict))
print('Recall:',recall_score(y_test ,dt_predict))
print('F1 score:',f1_score(y_test ,dt_predict))
print('Confusion matrix:',confusion_matrix(y_test ,dt_predict))
print('Classifiction report:',classification_report(y_test ,dt_predict))

#-----------Random forest----------

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
rf_predict=rf.predict(X_test)

print('-------random forest-----')
print('Accuracy:',accuracy_score(y_test,rf_predict))
print('Precision:',precision_score(y_test,rf_predict))
print('Recall:',recall_score(y_test,rf_predict))
print('F1 score:',f1_score(y_test,rf_predict))
print('Confusion matrix:',confusion_matrix(y_test,rf_predict))
print('Classification report:',classification_report(y_test,rf_predict))

importance = pd.Series(rf.feature_importances_, index=X.columns)

importance.sort_values().plot(kind="barh", figsize=(8,5))

plt.title("Feature Importance")
plt.savefig(
    r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#------------KNN-----------
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_predict=knn.predict(X_test)

print('\n -------KNN---------')
print('Accuracy:',accuracy_score(y_test,knn_predict))
print('Precision:',precision_score(y_test,knn_predict))
print('Recall:',recall_score(y_test,knn_predict))
print('F1 score:',f1_score(y_test,knn_predict))
print('Confusion Matrix:',confusion_matrix(y_test,knn_predict))
print('Classification report:',classification_report(y_test,knn_predict))

#------------SVM-----------

from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
svm_predict=svm.predict(X_test)

print('\n-----SVM------')
print('Acuracy',accuracy_score(y_test,svm_predict))
print('Precision',precision_score(y_test,svm_predict))
print('Recall',recall_score(y_test,svm_predict))
print('F1 score',f1_score(y_test,svm_predict))
print('Cofusion matrix',confusion_matrix(y_test,svm_predict))
print('Classification report',classification_report(y_test,svm_predict))

#-----------NAIVE BAYES---------
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
nb_predict=nb.predict(X_test)

print('\n-------- NAIVE BAYES----------')
print('Accuracy:',accuracy_score(y_test,nb_predict))
print('Precision:',precision_score(y_test,nb_predict))
print('Recall:',recall_score(y_test,nb_predict))
print('F1 score:',f1_score(y_test,nb_predict))
print('Confusion matrix:',confusion_matrix(y_test,nb_predict))
print('Classification report:',classification_report(y_test,nb_predict))

#--------Conparison  Table----------
comparison = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
        "Naive Bayes"
    ],

    "Accuracy":[
        accuracy_score(y_test,lr_predict),
        accuracy_score(y_test,dt_predict),
        accuracy_score(y_test,rf_predict),
        accuracy_score(y_test,knn_predict),
        accuracy_score(y_test,svm_predict),
        accuracy_score(y_test,nb_predict)
    ],

    "Precision":[
        precision_score(y_test,lr_predict),
        precision_score(y_test,dt_predict),
        precision_score(y_test,rf_predict),
        precision_score(y_test,knn_predict),
        precision_score(y_test,svm_predict),
        precision_score(y_test,nb_predict)
    ],

    "Recall":[
        recall_score(y_test,lr_predict),
        recall_score(y_test,dt_predict),
        recall_score(y_test,rf_predict),
        recall_score(y_test,knn_predict),
        recall_score(y_test,svm_predict),
        recall_score(y_test,nb_predict)
    ],

    "F1 Score":[
        f1_score(y_test,lr_predict),
        f1_score(y_test,dt_predict),
        f1_score(y_test,rf_predict),
        f1_score(y_test,knn_predict),
        f1_score(y_test,svm_predict),
        f1_score(y_test,nb_predict)
    ]

})

print(comparison)

#---------Acuracy comparision Graph -----------
plt.figure(figsize=(8,5))
plt.bar(comparison['Model'],comparison['Accuracy'])
plt.title('Classification Models Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.savefig(r"C:\Users\Awais\Documents\My_Course_AI_Bin\ML_Model\image\model_accuracy_comparison.png",dpi=300,
            bbox_inches="tight")
plt.show()

#------Best model---------

best_model=comparison.loc[comparison['Accuracy'].idxmax()]
print("\nBest Model")
print("Model Name :", best_model["Model"])
print("Accuracy   :", best_model["Accuracy"])