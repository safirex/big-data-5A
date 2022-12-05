# import projet.readFileHiggs as rf
import readFile as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_pca(data):
    pca = PCA(n_components=len(rf.get_column_labels()))
    pca.fit(np.transpose(data))
    return pca

n_red = 200
n_white = 200


test = rf.read_red(n_red)
test = test.append(rf.read_white(n_white))
print(test)

pca = create_pca(test)
data = pca.components_
print(data)

# # keep 2D data
# data = data[:2]
# data = test['fixed acidity']['volatile acidity']
# data  = data.append(test['volatile acidity'])
print(data)
names = ['red']*n_red + ['gray']*n_white
plt.figure(0)
plt.scatter(
    data[0],
    data[1],
    color=names
)
# plt.show()


cluster = KMeans(n_clusters=2,random_state=0).fit(data)
points = cluster.cluster_centers_
print(pd.DataFrame(points))
print(cluster.labels_)
print(len(cluster.labels_))
plt.scatter(
    points[0],
    points[1],
    color = 'black' 
)
plt.show()

# https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset