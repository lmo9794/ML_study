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
import numpy as np
from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


train_label = open('./train-labels-idx1-ubyte','rb')
train_image = open('./train-images-idx3-ubyte','rb')
test_label = open('./t10k-labels-idx1-ubyte','rb')
test_image = open('./t10k-images-idx3-ubyte','rb')

img = np.zeros((28,28))
lbl = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
d = 0
l = 0
index=0
s = train_image.read(16)    #헤더정보 (16byte) 8byte로 하면 짤림
l = train_label.read(8)     #헤더정보 (8byte)

k=0 #테스트용 index
#read mnist and show number
while True:    
    s = train_image.read(784) #784바이트씩 읽음
    l = train_label.read(1) #1바이트씩 읽음

    if not s:
        break;
    if not l:
        break;

    index = int(l[0])
    #print(k,":",index)

#unpack
    img = np.reshape( unpack(len(s)*'B',s), (28,28))
    lbl[index].append(img) #각 숫자영역별로 해당이미지를 추가
    k=k+1

#print(img)

plt.imshow(img,cmap = cm.binary) #binary형태의 이미지 설정
plt.show()

print(np.shape(lbl)) #label별로 잘 지정됬는지 확인

print("read done")

m_img = []

for i in range(0,10):
    m_img.append( np.mean(lbl[i],axis=0) )

for i in range(0,10):
    plt.imshow(m_img[i],cmap = cm.binary)
    plt.show()
    
    C = 10
# Create different classifiers.
classifiers =SVC(kernel='linear', C=C, probability=True,random_state=0)


#plt.plot(train_image,train_label,test_image,test_label)

#n_classifiers = len(classifiers)

SVC.fit(lbl, index)
for index in range(0,10) :


    y_pred = classifier.predict(test_image)
    accuracy = accuracy_score(test_image, test_label)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(test_image,test_label))
