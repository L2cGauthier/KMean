# K-mean clustering algorithm implementation

This project is a learning experiment. It's aimed at implementing one of the most simple clustering algorithm: the K-mean algorithm, in 2D, and to visualize how it works.

## The K-means algorithm

In machine learning, the k-means algorithm is an unsupervised learning used for clustering data, that's to say group observations based on their features to try to discover patterns.

The K-means algorithm takes as an input :
* A given a set of points of R<sup>n</sup>, which are our observation vectors of n features;
* An integer k, the number of clusters we are looking for.

The algorithm proceeds as follow:

```
Choose k data points at random as centroid candidates

While the association between data points and centroids keep changing

	For each point in the data set
	
		Compute its distance to all centroid candidates
		
		Associate that data point to the closest centroid candidate (using a distance measure like the euclidean distance)
	
	For each centroid candidate
	
		Calculate a new centroid candidate that is a vector which components are the mean of all associated data point

We know have our centroids, and their associated data points forming clusters around them
```

## Visualization

Using a the test script, we generated a random data set with 3 apparent clusters. Then using our implementation of the K-means algorithm, we obtain the following results.

![K-means visualization GIF](https://github.com/L2cGauthier/KMeans/blob/master/Example/Results/resultSummary.gif?raw=true
)

## Weaknesses of the k-means algorithm

* The user of k-means has to specify the number of centroids k
* As it relies on a distance measure, it is subjected to the curse of dimensionality
* It doesn't always work well on non-convex shaped clusters
* It doesn't handle well outliers because of the way centroids are calculated
* The initial choice of centroid candidates has an impact on the end result and might not be the best way to initialize the algorithm


