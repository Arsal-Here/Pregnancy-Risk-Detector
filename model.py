import joblib
from sklearn.model_selection import cross_val_score , train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,mean_absolute_error 
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
print(df.head)

#Making the categorical values to numerical
df['Risk Level'] = (df['Risk Level']=="High").astype(int)

#Setting the Target and Feature variables
cols = ['Age','Systolic BP','Diastolic','BS','Body Temp','BMI','Previous Complications','Preexisting Diabetes','Gestational Diabetes','Mental Health','Heart Rate','Risk Level']
y = df['Risk Level']
X = df[cols[:-1]]

#filling in NaN values with the mean values
for i in cols[:-1]:
  if X[i].isnull().any():
    X[i] = X[i].mean()

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=1)

NBmodel = GaussianNB()
NBmodel.fit(x_train,y_train)
predictions = NBmodel.predict(x_test)
mae = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error: ",mae)
print(predictions)
print(classification_report(y_test,predictions))

joblib.dump(NBmodel,'model.pkl')

