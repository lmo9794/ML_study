import numpy as np

import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train = np.load('./regression.npy')
test = np.load('./regression_test.npy')

X_train=np.c_[train[:,0]]
y_train=np.c_[train[:,1]]
X_test=np.c_[test[:,0]]
y_test=np.c_[test[:,1]]
y_train=y_train.astype('uint8')
y_test=y_test.astype('uint8')

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
