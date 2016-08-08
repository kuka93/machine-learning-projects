AIML-15: Machine Learning, Homework 1.
Principal Component Analysis and Naive Bayes Classification
November 16, 2015

General information. Problem solutions should be submitted in PDF format
in report style (no source code listings required). All reports must be
submitted before December 20 to the moodle (elearning) system. It is advised
to use Python as a programming language, but you can use any language of your
choice (at your own risk). In case you use Python, free Anaconda distribution
comes with all needed packages:
https://www.continuum.io/downloads
In particular, you might nd useful scikit-learn general machine learning
library and matplotlib plotting facilities. When in doubt, read the manual
and take a look at the large set of examples:
http://scikit-learn.org/stable/documentation.html
http://scikit-learn.org/stable/auto_examples/
http://matplotlib.org/examples/
Data preparation. In this homework you will work with the COIL-100 [1]
dataset of 100 visual object categories. The dataset contains  7000, 128128
colored images with subtracted background. Steps:
1. Download and unpack dataset from http://www.cs.columbia.edu/CAVE/
databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip.
2. Read raw pixels of all images for three classes of your choice. In python
you can do so by:
import Image
import numpy as np
img data = np . asar ray ( Image . open( '<path􀀀to􀀀image>' ) )
This will give you a 3-d array, where the last dimension species color of
a pixel in RGB.
1
3. Convert every image into a 49152-dimensional vector and prepare n49152
matrix X, where n is the number of images you have read. We refer to
the rows of X as examples and to columns as features. Next, prepare an
n-dimensional vector y holding ordinal labels of your image.
Note that in python you can get vectorial representation of your 3-d array
image array by:
x = img data . r a v e l ( )
Principal Component Visualization.
1. Standardize X (make each feature zero-mean and unit-variance).
2. Use Principal Component Analysis (PCA) to extract rst two principal
components from sample covariance matrix of X. Project X onto those
two components. You can do it in python by running:
from s k l e a rn . de compos i t ion import PCA
X t = PCA( 2 ) . f i t t r a n s f o rm (X)
3. Visualize X_t using scatter-plot with dierent colors standing for dierent
classes:
import ma tpl o t l ib . pyplot as p l t
p l t . s c a t t e r (X t [ : , 0 ] , X t [ : , 1 ] , c=y )
Repeat this exercise when considering third and fourth principal component,
and then tenth and eleventh. What do you notice? Justify your
answer from theoretical perspective behind PCA.
4. How would you decide on the number of principal components needed to
preserve data without much distortion?
Classication.
1. Write down formulation of Nave Bayes classier
by = arg max
y2f1;:::;kg
p(y j x1; : : : ; xd) ;
where by is a predicted label, k is the number of classes, fxigdi
=1 are examples,
p(x j y) is a Gaussian, and distribution of labels is uniform.
2. Split examples in X and y into training and testing set. You can use
train_test_split from sklearn.cross_validation package.
3. Train and test Nave Bayes classier with Gaussian class-conditional distribution.
You can use GaussianNB from package from sklearn.naive_bayes
for this purpose.
2
4. Repeat the splitting, training, and testing for the data projected onto rst
two principal components, then third and fourth principal components.
Compare results: what are your conclusions?
5. (Optional) Visualize decision boundaries of the classier.
References
[1] S. Nayar, S. A. Nene, and H. Murase. Columbia object image library
(coil 100). Department of Comp. Science, Columbia University, Tech. Rep.
CUCS-006-96, 1996.
