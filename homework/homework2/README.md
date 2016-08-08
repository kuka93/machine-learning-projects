==============
AIML-15: Machine Learning, Homework 2.
Support Vector Machine and Model Selection
==============
November 23, 2015


<b>Data preparation.</b> In this homework you will work with the MNIST [1] dataset composed from 10 classes of handwritten digits. The dataset contains  70000,
28  28 images. 
1 - Training Linear Support Vector Machine (SVM).
 Select the subset of X and y, which belongs only to classes 1 and 7.
 Standardize, shuffle, and split selected data into the training, validation, and testing sets as 50%; 20%; 30%.
 Train linear binary SVM, varying parameter C in the range of your choice, and plot it's accuracy on the validation set against every choice of C. In
Python you can use scikit-learn
 Which C will you use for the final classifier? Why?
 Train linear binary SVM once again, setting the best C. Test your classifier on the testing set and report obtained score.

<b>Training Multiclass Non-Linear SVM.</b>
 From X and y select examples which belong only to classes 0; 1; 2; 3; 4.
 Standardize, shue, and split selected data into the training and testing sets as 50=50% in a stratied way. Stratied means that all classes
should be represented in both training and testing sets, according to the splitting ratio. Consider label set y = f1; 1; 2; 2; 3; 3; 3; 3g, where example
of stratied splitting is , ytrain = f1; 2; 3; 3g, and ytest = f1; 2; 3; 3g, and a counter-example would be ytrain = f1; 3; 3; 3g, and ytest = f2; 2; 3; 3g.
Why is it important to do stratified splitting?
 Train a multiclass non-linear SVM with Gaussian kernel function in One-vs-All (OVA) way. You can train and get margin values of a binary non-
linear SVM. 
 How can you use it in for multiclass OVA classication? Write down a short explanation.
 Train multiclass non-linear SVM and report the accuracy on the testing set.
 Tune C and values of multiclass non-linear SVM using the grid search, to give the best performance. Remember, you cannot touch testing set during
tuning! Explain your steps for tuning these hyper-parameters. Finally, when you have tuned everything, report the accuracy of the classier on
the testing set. 

<i>
References
[1] Y. LeCun, C. Cortes, and C. JC Burges. The mnist database of handwritten digits, 1998.
</i>
