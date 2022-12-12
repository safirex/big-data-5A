# import projet.readFileHiggs as rf
import readFile as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_pca(data):
    pca = PCA(n_components=len(rf.get_column_labels()))
    pca.fit(np.transpose(data))
    return pca

n_red = 200
n_white = 200


test :pd.DataFrame = rf.read_red(n_red)
test = test.append(rf.read_white(n_white))
colors = ['red']*n_red + ['gray']*n_white

# pca = create_pca(test)
# data = pca.components_
data= test

plt.subplot(211)
plt.title("reality of wine type scattering")
data = test
fig2 = plt.scatter(
    data['fixed acidity'],
    data['volatile acidity'],
    color= colors
)

cluster = KMeans(n_clusters=2,random_state=0,n_init= 30).fit(data)
points = cluster.cluster_centers_
print(pd.DataFrame(points))
print("inertia",cluster.inertia_)
fig2= plt.scatter(
    points[:,0],
    points[:,1],
    color = 'black'
)

plt.subplot(212)
plt.title("kmean prediction of wine type scattering")

# data = data[['fixed acidity','volatile acidity']].to_numpy()
kmean = cluster.predict(data)
points = cluster.cluster_centers_

datas = data[['fixed acidity','volatile acidity']].to_numpy()

filtered_label0 = datas[kmean == 0]
filtered_label1 = datas[kmean == 1]

fig = plt.scatter(
    filtered_label0[:,0], 
    filtered_label0[:,1],
    color=['red']*len(filtered_label0)
)
fig = plt.scatter(
    filtered_label1[:,0], 
    filtered_label1[:,1],
    color=['gray']*len(filtered_label1)
) 
fig2= plt.scatter(
    points[:,0],
    points[:,1],
    color = 'black'
)
plt.show()



# elbow method for k-means clustering 
wcss = []

for i in range(1, 13):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 13), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

# show matrix of scatter over full every dimensions
data = test
# colors = np.array(colors)
# print(type(colors))
# data = data.merge(pd.DataFrame(colors,columns=['color']),how='cross')
# data.
# print(data)

fig = px.scatter_matrix(
    data,
    dimensions=data.columns,
    color = colors
)
fig.update_traces(diagonal_visible=False)
# fig.show()



cluster = KMeans(n_clusters=4,random_state=0,n_init= 30).fit(data)
points = cluster.cluster_centers_
# show matrix of scatter over full every dimensions
kmean= cluster.predict(data)
data = test
colors = ['red']*n_red + ['gray']*n_white
colors = ['red','gray','green','blue']
colors=  [colors[kmean[i]] for i in range(len(kmean))]


# data = data.merge(pd.DataFrame(colors,columns=['color']),how='cross')

fig = px.scatter_matrix(
    data,
    dimensions=data.columns,
    color = colors
)
fig.update_traces(diagonal_visible=False)
fig.show()









# from sklearn.metrics import confusion_matrix
# import seaborn  as sns

# wine_type_matrix =[ 0  if colors[i] =='red' else 1  for i in range(len(colors))  ]
# mat = confusion_matrix(wine_type_matrix, kmean)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=digits.target_names,
#             yticklabels=digits.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label');



# data = test[['total sulfur dioxide','volatile acidity']]
# pca = PCA(n_components=2)
# pca.fit(np.transpose(data))
# data = pca.components_

# plt.figure(0)
# plt.scatter(
#     data[0],
#     data[1],
#     color=colors
# )
# # plt.show()


# cluster = KMeans(n_clusters=2,random_state=0).fit(data)
# points = cluster.cluster_centers_
# print(pd.DataFrame(points))
# print(cluster.labels_)
# print(len(cluster.labels_))
# plt.scatter(
#     points[0],
#     points[1],
#     color = 'black' 
# )
# plt.show()

# https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset