# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PADMAVATHI M
RegisterNumber:  212223040141
*/
```
```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test date
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
## head():
![image](https://github.com/user-attachments/assets/4068f698-dfe6-40d2-90b3-c8b65512e799)

## tail():
![image](https://github.com/user-attachments/assets/e9a4032d-3e4e-46f2-a2df-52237168c219)

## X:
![image](https://github.com/user-attachments/assets/450ac0c5-c572-45ea-8c5e-272457a07ee1)

## Y:
![image](https://github.com/user-attachments/assets/7c8fd443-238a-4872-a82e-ddf76f4a0ad4)

## y_pred:
![image](https://github.com/user-attachments/assets/72049407-78ea-4e73-b797-7df60b05d983)

## y_test:
![image](https://github.com/user-attachments/assets/ea745e63-f93e-4dcf-9c91-18e89b1bc168)

![image](https://github.com/user-attachments/assets/a8025374-acef-4fdb-87dc-23c34f60b690)

![image](https://github.com/user-attachments/assets/bc328e94-46a8-45b2-9b11-42d2745d2ae6)

![image](https://github.com/user-attachments/assets/42256551-f43f-4905-b8fb-4e09c217d295)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
