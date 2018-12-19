# import requirements
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
# set the model and set the count of clustiung
kmn = KMeans(n_clusters=3)
kmn.fit(iris.data)
# predict
labels = kmn.predict(iris.data)
# select the two columns
xs = iris.data[:,1]
ys = iris.data[:,2]
centroids = kmn.cluster_centers_
# visualize the data
plt.scatter(xs, ys, c=labels)
plt.scatter(centroids[:,1],centroids[:,2],marker='x',s=150,alpha=0.5)
plt.show()
