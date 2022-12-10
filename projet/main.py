# import projet.readFileHiggs as rf
import readFile as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_pca(data):
    pca = PCA(n_components=len(rf.get_column_labels()))
    pca.fit(data)
    return pca

n_red = 50
n_white = 50


test = rf.read_red(n_red)
test = test.append(rf.read_white(n_white))
print(test)

pca = create_pca(test)
# pca = PCA(n_components= len(rf.get_column_labels())).fit(test)
data = pca.components_
print(pd.DataFrame(data))

# # keep 2D data
# data = data[:2]
data2 = test[['fixed acidity','volatile acidity']]
data2['fixed acidity']      = data2['fixed acidity'].astype('float')
data2['volatile acidity']   = data2['volatile acidity'].astype('float')

# print(data)
# print( data.to_numpy()[0])
colors = ['red']*n_red + ['gray']*n_white

print("ma data")
print( data2)

fig = plt.figure(0)

plt.scatter(
    data2['fixed acidity'],
    data2['volatile acidity'],
    color= colors
)

cluster = KMeans(n_clusters=2,random_state=0,n_init= 30).fit(data2)
points = cluster.cluster_centers_
print(pd.DataFrame(points))
print(cluster.labels_)
print("inertia",cluster.inertia_)
plt.scatter(
    points[:,0],
    points[:,1],
    color = 'black'
)
plt.show()










# https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html