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

n_red = 50
n_white = 50


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
    color=[names[t] for t in range(n_red+n_white)]
)
plt.show()


def kmean(data):
    pass