<b>
AIML-15: Machine Learning, Homework 1.
==============

Principal Component Analysis and Naive Bayes Classification
</b>
<i>November 16, 2015</i>


<b>Data preparation.</b> In this homework you will work with the COIL-100 [1] dataset of 100 visual object categories. The dataset contains  7000, 128x128 colored images with subtracted background. Steps:

1. Download and unpack dataset from http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip.

2. Read raw pixels of all images for three classes of your choice. 

3. Convert every image into a 49152-dimensional vector and prepare nx49152 matrix X, where n is the number of images you have read. We refer to the rows of X as examples and to columns as features. Next, prepare an n-dimensional vector y holding ordinal labels of your image. 


<b>Principal Component Visualization.</b>
1. Standardize X (make each feature zero-mean and unit-variance).

2. Use Principal Component Analysis (PCA) to extract first two principal components from sample covariance matrix of X. Project X onto those two components. 

3. Visualize X_t using scatter-plot with different colors standing for different classes. Repeat this exercise when considering third and fourth principal component, and then tenth and eleventh. What do you notice? Justify your answer from theoretical perspective behind PCA.

4. How would you decide on the number of principal components needed to preserve data without much distortion?

<b>Classification.</b>
1. Write down formulation of Naive Bayes classifer

2. Split examples in X and y into training and testing set. 

3. Train and test Naive Bayes classier with Gaussian class-conditional distribution. 

4. Repeat the splitting, training, and testing for the data projected onto first two principal components, then third and fourth principal components. Compare results: what are your conclusions?

5. (Optional) Visualize decision boundaries of the classifer.

<i>References
[1] S. Nayar, S. A. Nene, and H. Murase. Columbia object image library (coil 100). Department of Comp. Science, Columbia University, Tech. Rep. CUCS-006-96, 1996.
</i>
