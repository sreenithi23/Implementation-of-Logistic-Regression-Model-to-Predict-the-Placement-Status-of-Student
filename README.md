# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```

1 Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:sreenithi.E 
RegisterNumber: 212223220109 
*/
```
```

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
`



## Output:
Placement Data:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/9a56d86b-f790-403b-a584-44467edf4342)

Salary Data:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/d55c8659-0821-42d6-989a-51ce01146a2c)

Checking the null() function:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/f4bb5b4e-66fa-4535-83ec-b998a0ad23ef)

Data Duplicate:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/35e2cdcc-4cd4-4069-87d9-66e17897d2bc)

Print Data:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/fd28a395-b2fd-4e57-8f7a-bbf70690ebdc)

Data-Status:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/2c4084a7-d10d-48c3-bd28-58f206f93514)

Y_prediction array:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/4fa15bb4-5500-4985-bbf2-adc8977137d6)

Accuracy value:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/86c98c3c-3132-4eb8-a140-d6bd291e1042)

Confusion array:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/be2054cf-1890-402a-aeb5-32f360c9c76f)

Classification Report:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/715f69ec-618e-4db2-9d2d-d3e68291ab5b)

Prediction of LR:

![image](https://github.com/sreenithi23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147017600/5313cbab-7adb-4cf6-b84f-e15037376ecd)












## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
