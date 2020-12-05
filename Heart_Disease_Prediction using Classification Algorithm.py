#### Introduction:
##We have dataset which classified if person have heart disease or not according to feature in it.
##Using this dataset we create model which tries to predict the person have heart disease or not.

##Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

##Loading dataset
data = pd.read_csv("heart.csv")
print(data.head())
print(data.shape)
print(data['target'].unique())

#Now checking missing values in dataset
print(data.isnull().sum())

## We can see that there is no missing values in dataset

print(data.columns)

print(data.info())

# Lets see the statistical view of the dataset
print(data.describe())
print(data['target'].value_counts(normalize =True))
##We drop unwanted columns from dataset
print(data.drop(columns= ['slope', 'ca'], axis = 1, inplace = True))
print(data.dtypes)
col= ['age','sex','cp',"trestbps","chol",'fbs', 'restecg', 'exang', 'thalach']


##Splitting dataset into indepenent variable and dependent varible
X = data[col]
y = data['target']

print(X.head())
print(y.head())

#Now splitting dataset  into training data and testing data
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(x_train.shape, x_test.shape)


from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(n_estimators=30)

print(random_clf.fit(x_train, y_train))

random_predict = random_clf.predict(x_test)


##Let's check the performanc of the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
random_accuracy = accuracy_score(y_test, random_predict)
print(random_accuracy)

print(confusion_matrix(y_test, random_predict))

print(classification_report(y_test, random_predict))

import joblib
joblib.dump(random_clf, "heart_model.pkl")
