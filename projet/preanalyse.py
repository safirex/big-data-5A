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



test :pd.DataFrame = rf.read_red(n_red)
data = test.append(rf.read_white(n_white))
colors = ['red']*n_red + ['gray']*n_white

fig = px.scatter_matrix(
    data,
    dimensions=data.columns,
    color = colors
)
fig.update_traces(diagonal_visible=False)
fig.show()


