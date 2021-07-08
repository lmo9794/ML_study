import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
train = np.load('./regression.npy')
test = np.load('./regression_test.npy')
X_train=np.c_[train[:,0]]
y_train=np.c_[train[:,1]]
X_test=np.c_[test[:,0]]
y_test=np.c_[test[:,1]]
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train,  color='black')
plt.show()
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
pred = lin_reg.predict(X_test)
accuracy_score = lin_reg.score(X_train,y_train)
print('Model Accuracy: ', accuracy_score)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, pred, color='blue', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
