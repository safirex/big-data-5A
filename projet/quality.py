import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import readFile as rf


# data = data.merge(pd.DataFrame(colors,columns=['color']),how='cross')

def create_pca(data):
    pca = PCA(n_components=len(rf.get_column_labels()))
    pca.fit(np.transpose(data))
    return pca

n_red = 200
n_white = 200

colors = ['red']*n_red + ['gray']*n_white
colors = ['red','gray','green','blue']

test :pd.DataFrame = rf.read_red(n_red)
datas :pd.DataFrame= test.append(rf.read_white(n_white))
cluster = KMeans(n_clusters=4,random_state=0,n_init= 30).fit(datas)

kmean = cluster.predict(datas)
colors=  [colors[kmean[i]] for i in range(len(kmean))]

ax = plt.figure().add_subplot(projection='3d')
data = datas[['total sulfur dioxide','ph','quality']].to_numpy()
ax.scatter(
    xs= data[:,0],
    ys= data[:,1],
    zs= data[:,2],
    c=colors,
)
plt.xlabel('total sulfur dioxide')
plt.ylabel('ph')
# plt.zlabel('quality')

plt.title('observation de la répartition des cluster sulfure/ph en fonction de la quality')
plt.show()




'''   second graph'''


ax = plt.figure().add_subplot(projection='3d')
columns = ['ph','density','quality']
data = datas[columns].to_numpy()
ax.scatter(
    xs= data[:,0],
    ys= data[:,1],
    zs= data[:,2],
    c=colors,
)
plt.xlabel(columns[0])
plt.ylabel(columns[1])
# plt.zlabel('quality')

plt.title('observation de la répartition des cluster %s/%s en fonction de la %s'%(columns[0],columns[1],columns[2]))
plt.show()


'''  3rd graphe  '''

ax = plt.figure().add_subplot(projection='3d')
columns = ['volatile acidity','fixed acidity','free sulfur dioxide']
data = datas[columns].to_numpy()
ax.scatter(
    xs= data[:,0],
    ys= data[:,1],
    zs= data[:,2],
    c=colors,
)
plt.xlabel(columns[0])
plt.ylabel(columns[1])
# plt.zlabel('quality')

plt.title('observation de la répartition des cluster %s/%s en fonction de la %s'%(columns[0],columns[1],columns[2]))
plt.show()




'''  4e graphe  '''

ax = plt.figure().add_subplot(projection='3d')
columns = ['free sulfur dioxide','total sulfur dioxide','quality']
data = datas[columns].to_numpy()
ax.scatter(
    xs= data[:,0],
    ys= data[:,1],
    zs= data[:,2],
    c=colors,
)
plt.xlabel(columns[0])
plt.ylabel(columns[1])
# plt.zlabel('quality')

plt.title('observation de la répartition des cluster %s/%s en fonction de la %s'%(columns[0],columns[1],columns[2]))
plt.show()



ax = plt.figure().add_subplot(projection='3d')


pca = PCA(n_components=12)
pca = pca.fit(datas)
data = pca.components_
ax.scatter(
    xs= data[:,0],
    ys= data[:,1],
    zs= data[:,2],
)
plt.show()
