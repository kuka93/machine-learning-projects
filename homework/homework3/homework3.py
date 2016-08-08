__author__ = 'Sara'

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab


mnist = fetch_mldata('MNIST original')
X = mnist.data.astype(np.float64)  # 70000 by 784 matrix of instances
y = mnist.target  # 70000 vector of labels


# CLASS 0 (200 examples per class)
target_class0 = y[0:500]
data_class0 = X[0:500, :]

# CLASS 1 (200 examples per class)
target_class1 = y[5923:6423]
data_class1 = X[5923:6423, :]

# CLASS 2 (200 examples per class)
target_class2 = y[12665:13165]
data_class2 = X[12665:13165, :]

# CLASS 3
target_class3 = y[18623:19123]
data_class3 = X[18623:19123, :]

# CLASS 4
target_class4 = y[24754:25254]
data_class4 = X[24754:25254, :]

X2 = np.concatenate((data_class0, data_class1, data_class2, data_class3, data_class4), axis=0)
y2 = np.concatenate((target_class0, target_class1, target_class2, target_class3, target_class4), axis=0)

print "X2", X2.shape
print "y2", y2.shape

# --------  THIRD PART  --------
from sklearn.mixture import GMM

accuracies = []

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3)

X_train_0, X_test_0 = train_test_split(data_class0, test_size=0.3)
X_train_1, X_test_1 = train_test_split(data_class1, test_size=0.3)
X_train_2, X_test_2 = train_test_split(data_class2, test_size=0.3)
X_train_3, X_test_3 = train_test_split(data_class3, test_size=0.3)
X_train_4, X_test_4 = train_test_split(data_class4, test_size=0.3)

for j in range(2, 6):  # varying number of components in the range 2,3,4,5

    #CLASS 0
    clf0 = GMM(j, 'diag')
    clf0.fit(X_train_0)
    score_samples0 = clf0.score_samples(X_test)
    logprob0 = score_samples0[0]

    responsibilities0 = score_samples0[1]
    #print responsibilities0

    #CLASS1
    clf1 = GMM(j, 'diag')
    clf1.fit(X_train_1)
    score_samples1 = clf1.score_samples(X_test)
    logprob1 = score_samples1[0]
    responsibilities1 = score_samples1[1]
    #print responsibilities1

    #CLASS2
    clf2 = GMM(j, 'diag')
    clf2.fit(X_train_2)
    score_samples2 = clf2.score_samples(X_test)
    logprob2 = score_samples2[0]
    responsibilities2 = score_samples2[1]
    #print responsibilities2

    #CLASS3
    clf3 = GMM(j, 'diag')
    clf3.fit(X_train_3)
    score_samples3 = clf3.score_samples(X_test)
    logprob3 = score_samples3[0]
    responsibilities3 = score_samples3[1]
    #print responsibilities3

    #CLASS4
    clf4 = GMM(j, 'diag')
    clf4.fit(X_train_4)
    score_samples4 = clf4.score_samples(X_test)
    logprob4 = score_samples4[0]
    responsibilities4 = score_samples4[1]
    #print responsibilities4

    logprob_matrix = np.column_stack((logprob0, logprob1))
    logprob_matrix = np.column_stack((logprob_matrix, logprob2))
    logprob_matrix = np.column_stack((logprob_matrix, logprob3))
    logprob_matrix = np.column_stack((logprob_matrix, logprob4))
    expectedLabelVec = np.empty(300)

    for i in range(300):
        maxlogprob = logprob_matrix[i][0]
        expectedLabel = 0
        for q in range(1, 5):
            if maxlogprob < logprob_matrix[i][q]:
                maxlogprob = logprob_matrix[i][q]
                expectedLabel = q
        expectedLabelVec[i] = expectedLabel

    misclassifications = 0
    for i in range(300):
        if expectedLabelVec[i] != y_test[i]:
            misclassifications += 1

    print "MISCLASSIFICATIONS " + str(j)+" " + str(misclassifications)

#Select number of components with train-validation-test
X_trainCV, X_valCV, y_trainCV, y_valCV = train_test_split(X_train, y_train, test_size=0.14)

X_trainCV_0, X_valCV_0 = train_test_split(X_train_0, test_size=0.3)
X_trainCV_1, X_valCV_1 = train_test_split(X_train_1, test_size=0.3)
X_trainCV_2, X_valCV_2 = train_test_split(X_train_2, test_size=0.3)
X_trainCV_3, X_valCV_3 = train_test_split(X_train_3, test_size=0.3)
X_trainCV_4, X_valCV_4 = train_test_split(X_train_4, test_size=0.3)

min_misclassifications = 300
k_best = 0
for j in range(2, 6):  # varying number of components in the range 2,3,4,5

    #CLASS0
    clf0 = GMM(j, 'diag')
    clf0.fit(X_trainCV_0)
    score_samples0 = clf0.score_samples(X_valCV)
    logprob0_2 = score_samples0[0]

    responsibilities0 = score_samples0[1]
    #print responsibilities0

    #CLASS1
    clf1 = GMM(j, 'diag')
    clf1.fit(X_trainCV_1)
    score_samples1 = clf1.score_samples(X_valCV)
    logprob1_2 = score_samples1[0]
    responsibilities1 = score_samples1[1]
    #print responsibilities1

    #CLASS2
    clf2 = GMM(j, 'diag')
    clf2.fit(X_trainCV_2)
    score_samples2 = clf2.score_samples(X_valCV)
    logprob2_2 = score_samples2[0]
    responsibilities2 = score_samples2[1]
    #print responsibilities2

    #CLASS3
    clf3 = GMM(j, 'diag')
    clf3.fit(X_trainCV_3)
    score_samples3 = clf3.score_samples(X_valCV)
    logprob3_2 = score_samples3[0]
    responsibilities3 = score_samples3[1]
    #print responsibilities3

    #CLASS4
    clf4 = GMM(j, 'diag')
    clf4.fit(X_trainCV_4)
    score_samples4 = clf4.score_samples(X_valCV)
    logprob4_2 = score_samples4[0]
    responsibilities4 = score_samples4[1]
    #print responsibilities4

    #print logprob0_2[0], logprob1_2[0], logprob2_2[0], logprob3_2[0], logprob4_2[0]
    logprob_matrix2 = np.column_stack((logprob0_2, logprob1_2))
    logprob_matrix2 = np.column_stack((logprob_matrix2, logprob2_2))
    logprob_matrix2 = np.column_stack((logprob_matrix2, logprob3_2))
    logprob_matrix2 = np.column_stack((logprob_matrix2, logprob4_2))
    expectedLabelVec = np.empty(len(y_valCV))

    #print logprob_matrix2[0]

    for i in range(len(X_valCV)):
        maxlogprob = logprob_matrix2[i][0]
        expectedLabel = 0
        for q in range(1, 5):
            if maxlogprob < logprob_matrix2[i][q]:
                maxlogprob = logprob_matrix2[i][q]
                expectedLabel = q
        expectedLabelVec[i] = expectedLabel

    misclassifications = 0
    for i in range(len(X_valCV)):
        if expectedLabelVec[i] != y_valCV[i]:
            misclassifications += 1
    if misclassifications<min_misclassifications:
        min_misclassifications = misclassifications
        k_best = j
    print "MISCLASSIFICATIONS " + str(j)+" " + str(misclassifications)

#with best k

#CLASS0
clf0 = GMM(k_best, 'diag')
clf0.fit(X_trainCV_0)
score_samples0_3 = clf0.score_samples(X_test)
logprob0_3 = score_samples0_3[0]

#CLASS1
clf1 = GMM(k_best, 'diag')
clf1.fit(X_trainCV_1)
score_samples1_3 = clf1.score_samples(X_test)
logprob1_3 = score_samples1_3[0]

#CLASS2
clf2 = GMM(k_best, 'diag')
clf2.fit(X_trainCV_2)
score_samples2_3 = clf2.score_samples(X_test)
logprob2_3 = score_samples2_3[0]

#CLASS3
clf3 = GMM(k_best, 'diag')
clf3.fit(X_trainCV_3)
score_samples3_3 = clf3.score_samples(X_test)
logprob3_3 = score_samples3_3[0]

#CLASS4
clf4 = GMM(k_best, 'diag')
clf4.fit(X_trainCV_4)
score_samples4_3 = clf4.score_samples(X_test)
logprob4_3 = score_samples4_3[0]

logprob_matrix3 = np.column_stack((logprob0_3, logprob1_3))
logprob_matrix3 = np.column_stack((logprob_matrix3, logprob2_3))
logprob_matrix3 = np.column_stack((logprob_matrix3, logprob3_3))
logprob_matrix3 = np.column_stack((logprob_matrix3, logprob4_3))
expectedLabelVec = np.empty(len(y_test))

for i in range(len(y_test)):
    maxlogprob = logprob_matrix3[i][0]
    expectedLabel = 0
    for q in range(1, 5):
        if maxlogprob < logprob_matrix3[i][q]:
            maxlogprob = logprob_matrix3[i][q]
            expectedLabel = q
    expectedLabelVec[i] = expectedLabel

misclassifications = 0
for i in range(len(y_test)):
    if expectedLabelVec[i] != y_test[i]:
        misclassifications += 1

# print expectedLabelVec
# print y_test
print "Best k: ", k_best
print "MISCLASSIFICATIONS with best k" + " " + str(misclassifications)
#print y_test[0]
#print logprob0[0]
print ((len(y_test) - misclassifications) * 100)/len(y_test)
