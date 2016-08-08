AIML-15: Machine Learning, Homework 3.
==============
K-Means, GMM+EM, and Clustering Evaluation
==============
December 1, 2015

<b>Data preparation.</b> In this homework you will work with the MNIST [1] dataset composed from 10 classes of handwritten digits. The dataset contains 70000,
28 x 28 images. 

<b>Clustering with K-Means.</b>
 Select the subset of X and y, which belongs only to classes f0; 1; 2; 3; 4g, with 200 examples per class.
 Cluster X using K-Means into 5 clusters. 
 Plot obtained cluster centroids as images. You can use pylab.imshow().
 Repeat clustering and visualization for 3 clusters and 10 clusters. Which characteristic of the data is captured by the centroids?

<b>Clustering with GMM/EM, and Performance Evaluation.</b>
 Cluster X multiple times using GMM when number of clusters varies is in {2; 3; : : : ; 10}. 
 Compute cluster purity for every choice of number of clusters, and plot number of clusters against the purity.
 Explain your observation.

<b>Classifying with GMM/EM.</b>
In this assignment you're asked to construct a classifier through generative modeling, that is, every class will be modeled by
a mixture of Gaussians. Guidelines:
 For every class in X, fit a GMM model as in previous problem.
 For every point in the testing set, obtain the log-likelihood of that a point, and decide the label as index of the GMM with the highest log-likelihood.
 Report performance of this classier on the testing set, varying number of components in each mixture in {2; 3; 4; 5}.
 (Optional) Select number of components K using validation set or cross-validation.
 (Optional) Compare to non-linear SVM. 

<i>
References
[1] Y. LeCun, C. Cortes, and C. JC Burges. The mnist database of handwritten digits, 1998.
</i>
