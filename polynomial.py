import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())

train = np.load('../data/regression.npy')
test = np.load('../data/regression_test.npy')

X_train=np.c_[train[:,0]]
y_train=np.c_[train[:,1]]
X_test=np.c_[test[:,0]]
y_test=np.c_[test[:,1]]

pipeline.fit(np.array(X_train), y_train)
y_pred=pipeline.predict(X_test)

df = pd.DataFrame({'x': X_test[:,0], 'y': y_pred[:,0]})
df.sort_values(by='x',inplace = True)
points = pd.DataFrame(df).to_numpy()

plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X_test,y_test, color="black")
plt.show()

accuracy_score = pipeline.score(X_train,y_train)
print('Model Accuracy: ', accuracy_score)