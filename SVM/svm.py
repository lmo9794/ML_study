import numpy as np
import sys
import os
from array import array
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

train_image = np.fromfile('./train-images-idx3-ubyte', dtype = np.uint8)
train_label = np.fromfile('./train-labels-idx1-ubyte',dtype = np.uint8)
test_image = np.fromfile('./t10k-images-idx3-ubyte',dtype = np.uint8)
test_label = np.fromfile('./t10k-labels-idx1-ubyte',dtype = np.uint8)


train_image = np.delete(train_image, range(16))
train_image = train_image.reshape(-1, 784)
train_label = np.delete(train_label, range(8))

test_image = np.delete(test_image, range(16))
test_image = train_image.reshape(-1, 784)
test_label = np.delete(test_label, range(8))

C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
}

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(train_image, np.ravel(train_label))

    y_pred = classifier.predict(test_image)
    accuracy = accuracy_score(test_label, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(test_label,y_pred))
