import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import classification_report

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

scaler = MinMaxScaler()

# Fit og trans på ikke test
# Kan bruke fit først også trans
X_train_scaled = scaler.fit_transform(X_train)

# Transform på test
X_test_scaled = scaler.transform(X_test)

#KNN = Nomral = 5
neigth = KNeighborsClassifier()

# Fit med bruk av scale
neigth.fit(X_train_scaled, y_train)

# Pred
y_pred = neigth.predict(X_test_scaled)


target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))
