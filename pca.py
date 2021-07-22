#%%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import time

train_image = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
train_image = np.delete(train_image, range(16))
train_image = train_image.reshape(-1, 784)
print(train_image.shape)

train_label = np.fromfile('train-labels-idx1-ubyte',dtype = np.uint8)
train_label = np.delete(train_label, range(8))
print(train_label.shape)

test_image = np.fromfile('t10k-images-idx3-ubyte',dtype = np.uint8)
test_image = np.delete(test_image, range(16))
test_image = test_image.reshape(-1, 784)
print(test_image.shape)

test_label = np.fromfile('t10k-labels-idx1-ubyte',dtype = np.uint8)
test_label = np.delete(test_label, range(8))
print(test_label.shape)


classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
features = ['pixel' + str(i+1) for i in range(test_image.shape[1]) ]
pca_df = pd.DataFrame(test_image, columns=features)
pca_df['label'] = test_label

results = []
# Loop through all label
for i in range(pca_df.shape[0]):
    # Extract the label for comparison
    if pca_df['label'][i] == 0:
        # Save meaningful label to the results
        results.append('T-shirt/top')
    # Following the same code pattern as the one above
    elif pca_df['label'][i] == 1:
        results.append('Trouser')
    elif pca_df['label'][i] == 2:
        results.append('Pullover')
    elif pca_df['label'][i] == 3:
        results.append('Dress')
    elif pca_df['label'][i] == 4:
        results.append('Coat')
    elif pca_df['label'][i] == 5:
        results.append('Sandal')
    elif pca_df['label'][i] == 6:
        results.append('Shirt')
    elif pca_df['label'][i] == 7:
        results.append('Sneaker')
    elif pca_df['label'][i] == 8:
        results.append('Bag')
    elif pca_df['label'][i] == 9:
        results.append('Ankle boot')
    else:
        print("The dataset contains an unexpected label {}".format(pca_df['label'][i]))

# Create a new column named result which has all meaningful results        
pca_df['result'] = results

pca = PCA(n_components=3)
pca_result = pca.fit_transform(pca_df[features].values)

pca.fit(test_image)
pca_train_data = pca.transform(train_image)
pca_test_data = pca.transform(test_image)

pca_df['First Dimension'] = pca_test_data[:,0]
pca_df['Second Dimension'] = pca_test_data[:,1] 
pca_df['Third Dimension'] = pca_test_data[:,2]

graph = plt.figure(figsize=(16,10)).gca(projection='3d')
graph.scatter(
    xs=pca_df["First Dimension"], 
    ys=pca_df["Second Dimension"], 
    zs=pca_df["Third Dimension"], 
    c=pca_df["label"], 
    cmap='tab10'
)
graph.set_xlabel('First Dimension')
graph.set_ylabel('Second Dimension')
graph.set_zlabel('Third Dimension')

plt.show()



