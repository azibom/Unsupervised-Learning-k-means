# Unsupervised-Learning-k-means
k-means?

#### k-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

now we try to implement it :sunglasses:

```python
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
```
(if you don't run it yourself you can't understand it completely)

I hope this article will be useful to you.
