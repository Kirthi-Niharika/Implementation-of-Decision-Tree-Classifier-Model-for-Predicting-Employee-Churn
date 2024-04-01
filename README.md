# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/809374ce-c409-4faa-b118-883f25c37c5e)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/aa27228a-89b2-48c4-8708-7624acd19ffc)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/6f49a71c-4b7c-444d-b799-2fb26ad95469)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/7830b55a-e949-489b-9959-0e991e918aa0)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/8aa57d94-94b5-4749-8e04-8387de5f21db)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/d3ba486f-ca89-4daf-82cf-318a9cf7a883)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/c6b832b7-bcb1-480f-a5ca-f6dca84d91ba)

![image](https://github.com/Kirthi-Niharika/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/114135005/a9c79ba8-eb0d-43f9-8cd8-29c05276de01)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
