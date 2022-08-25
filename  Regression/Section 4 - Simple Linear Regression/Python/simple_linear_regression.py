# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #trains model on x and y test set

# Predicting the Test set results
regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red') #scattering dots of x and y train set
plt.plot(X_train, regressor.predict(X_train), color='blue') #drawing line of regression 
plt.title('training set') 
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red') #scattering dots of X and y test cordinates
plt.plot(X_test, regressor.predict(X_test), color='blue') #plotting regression test line
plt.title('test set') 
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()