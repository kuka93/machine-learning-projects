__author__ = 'Francesco'

from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


mnist = fetch_mldata('MNIST original')
X = mnist.data.astype(np.float64)  # 70000 by 784 matrix of instances
y = mnist.target  # 70000 vector of labels



"""
# Extracting elements of classes
"""

# CLASS 1
target_class1_1 = y[5923:12665]
target_class1_2 = y[60980:62115]
target_class1 = np.concatenate((target_class1_1, target_class1_2), axis=0)
# print target_class1.shape
data_class1_1 = X[5923:12665, :]
data_class1_2 = X[60980:62115, :]
data_class1 = np.concatenate((data_class1_1, data_class1_2), axis=0)
# print data_class1.shape

# CLASS 7
target_class7_1 = y[41935:48200]
target_class7_2 = y[66989:68017]
target_class7 = np.concatenate((target_class7_1, target_class7_2), axis=0)
# print target_class7.shape
data_class7_1 = X[41935:48200, :]
data_class7_2 = X[66989:68017, :]
data_class7 = np.concatenate((data_class7_1, data_class7_2), axis=0)
# print data_class7.shape


# CLASS 0
# 0/5922 60000/60979
target_class0_1 = y[0:5922]
target_class0_2 = y[60000:60979]
target_class0 = np.concatenate((target_class0_1, target_class0_2), axis=0)
# print target_class0.shape
data_class0_1 = X[0:5922, :]
data_class0_2 = X[60000:60979, :]
data_class0 = np.concatenate((data_class0_1, data_class0_2), axis=0)


# CLASS 2
# 12665/18622 62115/63146
target_class2_1 = y[12665:18622]
target_class2_2 = y[62115:63146]
target_class2 = np.concatenate((target_class2_1, target_class2_2), axis=0)
# print target_class2.shape
data_class2_1 = X[12665:18622, :]
data_class2_2 = X[62115:63146, :]
data_class2 = np.concatenate((data_class2_1, data_class2_2), axis=0)

# CLASS 3
# 18623/24753 63147/64156
target_class3_1 = y[18623:24753]
target_class3_2 = y[63147:64156]
target_class3 = np.concatenate((target_class3_1, target_class3_2), axis=0)
# print target_class2.shape
data_class3_1 = X[18623:24753, :]
data_class3_2 = X[63147:64156, :]
data_class3 = np.concatenate((data_class3_1, data_class3_2), axis=0)

# CLASS 4
# 24754/30595 64157/65138
target_class4_1 = y[24754:30595]
target_class4_2 = y[64157:65138]
target_class4 = np.concatenate((target_class4_1, target_class4_2), axis=0)
# print target_class4.shape
data_class4_1 = X[24754:30595, :]
data_class4_2 = X[64157:65138, :]
data_class4 = np.concatenate((data_class4_1, data_class4_2), axis=0)

'''
# FIRST PART

# Standardize, shuffle and split data into training, testing and validation set (50-30-20)
scaler = StandardScaler()
data_class1_std = scaler.fit_transform(data_class1)
data_class7_std = scaler.fit_transform(data_class7)

X_2 = np.concatenate((data_class1_std, data_class7_std), axis=0)
y_2 = np.concatenate((target_class1, target_class7), axis=0)


for i in range(2):
    sss = ShuffleSplit(len(y_2), n_iter=10, test_size=0.5)
    for train_index, test_index in sss:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_int = X_2[train_index], X_2[test_index]
        y_train, y_int = y_2[train_index], y_2[test_index]
    X_validation, X_test, y_validation, y_test = train_test_split(X_int, y_int, train_size=0.4, random_state=0)


    # C_range = np.array([10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e5, 10e6, 10e7, 10e8, 10e9])
    # C_range = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    C_range = np.array([0.001, 0.01, 0.1, 1.0, 2.0, 10.0, 50.0])

    accuracies = []

    for c in C_range:
        cls = LinearSVC(C=c)
        cls.fit(X_train, y_train)  # Train on X-train, y_train
        # accuracy = cls.score(X_test, y_test)  # Test on X_test, y_test
        accuracy = cls.score(X_validation, y_validation)  # Test on X_test, y_test
        accuracies.append(accuracy)
        print c, accuracy

    max_acc = max(accuracies)
    ind = accuracies.index(max_acc)
    best_c = C_range[ind]

    cls = LinearSVC(C=best_c)
    cls.fit(X_train, y_train)  # Train on X-train, y_train
    accuracy = cls.score(X_test, y_test)  # Test on X_test, y_test
    print "Best C:", best_c
    print "Accuracy on test set with best C: ", accuracy

    

    """
    import pylab
    pylab.plot(C_range, accuracies)
    pylab.ylim([0.98, 1.0])
    pylab.show()
    """
    from matplotlib import pyplot as plt
    plt.plot(C_range, accuracies)
    plt.xlabel('C range')
    plt.ylabel('Accuracies')
    plt.ylim([0.9850, 0.9990])
    filename = "plot"+str(i+1)+".png"
    plt.savefig(filename)
    plt.close()
'''

# SECOND PART
# CLASS 0 (200 examples per class)
target_class0 = y[0:200]
data_class0 = X[0:200, :]

# CLASS 1 (200 examples per class)
target_class1 = y[5923:6123]
data_class1 = X[5923:6123, :]

# CLASS 2 (200 examples per class)
target_class2 = y[12665:12865]
data_class2 = X[12665:12865, :]

# CLASS 3 (200 examples per class)
target_class3 = y[18623:18823]
data_class3 = X[18623:18823, :]

# CLASS 4 (200 example per class)
target_class4 = y[24754:24954]
data_class4 = X[24754:24954, :]


scaler = StandardScaler()
data_class0_std = scaler.fit_transform(data_class0)
data_class1_std = scaler.fit_transform(data_class1)
data_class2_std = scaler.fit_transform(data_class2)
data_class3_std = scaler.fit_transform(data_class3)
data_class4_std = scaler.fit_transform(data_class4)


X_new = np.concatenate((data_class0_std, data_class1_std, data_class2_std, data_class3_std, data_class4_std), axis=0)
y_new = np.concatenate((target_class0, target_class1, target_class2, target_class3, target_class4), axis=0)

print X_new.shape
print y_new.shape

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(y_new, test_size=0.5)
for train_index, test_index in sss:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X2train, X2test = X_new[train_index], X_new[test_index]
    y2train, y2test = y_new[train_index], y_new[test_index]


# class 0 vs ALL
print "CLASS 0 VS ALL"
import copy
y_train_temp = copy.deepcopy(y2train)
for i in range(len(y2train)):
    if y_train_temp[i] == 0.0:
        y_train_temp[i] = 1.0
    else:
        y_train_temp[i] = -1.0
y_test_temp = copy.deepcopy(y2test)
for i in range(len(y2train)):
    if y_test_temp[i] == 0.0:
        y_test_temp[i] = 1.0
    else:
        y_test_temp[i] = -1.0


from sklearn import grid_search
from sklearn.svm import SVC
parameters = {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y_train_temp)


print clf.best_params_
cls = SVC(kernel='rbf', C=10.0, gamma=0.001)
cls.fit(X2train, y_train_temp)  # Train on X_train, y_train
# Margin (decision) values of classifier
margins0 = cls.decision_function(X2test)
print "score 0 vs ALL", cls.score(X2test, y_test_temp)
pred0 = cls.predict(X2test)


# class 1 vs ALL
print "CLASS 1 VS ALL"

y_train_temp = copy.deepcopy(y2train)
for i in range(len(y2train)):
    if y_train_temp[i] == 1.0:
        y_train_temp[i] = 1.0
    else:
        y_train_temp[i] = -1.0
y_test_temp = copy.deepcopy(y2test)
for i in range(len(y2train)):
    if y_test_temp[i] == 1.0:
        y_test_temp[i] = 1.0
    else:
        y_test_temp[i] = -1.0

from sklearn import grid_search
from sklearn.svm import SVC
parameters = {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y_train_temp)


print clf.best_params_
cls = SVC(kernel='rbf', C=10.0, gamma=0.001)
cls.fit(X2train, y_train_temp)  # Train on X_train, y_train
margins1 = cls.decision_function(X2test)
print "score 1 vs ALL", cls.score(X2test, y_test_temp)
pred1 = cls.predict(X2test)



# class 2 vs ALL
print "CLASS 2 VS ALL"

y_train_temp = copy.deepcopy(y2train)
for i in range(len(y2train)):
    if y_train_temp[i] == 2.0:
        y_train_temp[i] = 1.0
    else:
        y_train_temp[i] = -1.0
y_test_temp = copy.deepcopy(y2test)
for i in range(len(y2train)):
    if y_test_temp[i] == 2.0:
        y_test_temp[i] = 1.0
    else:
        y_test_temp[i] = -1.0

parameters = {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y_train_temp)


print clf.best_params_
cls = SVC(kernel='rbf', C=10.0, gamma=0.001)
cls.fit(X2train, y_train_temp)  # Train on X_train, y_train
# Margin (decision) values of classifier
margins2 = cls.decision_function(X2test)
print "score 2 vs ALL", cls.score(X2test, y_test_temp)
pred2 = cls.predict(X2test)


# class 3 vs ALL
print "CLASS 3 VS ALL"
y_train_temp = copy.deepcopy(y2train)
for i in range(len(y2train)):
    if y_train_temp[i] == 3.0:
        y_train_temp[i] = 1.0
    else:
        y_train_temp[i] = -1.0
y_test_temp = copy.deepcopy(y2test)
for i in range(len(y2train)):
    if y_test_temp[i] == 3.0:
        y_test_temp[i] = 1.0
    else:
        y_test_temp[i] = -1.0

parameters = {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y_train_temp)


print clf.best_params_
cls = SVC(kernel='rbf', C=10.0, gamma=0.001)
cls.fit(X2train, y_train_temp)  # Train on X_train, y_train
# Margin (decision) values of classifier
margins3 = cls.decision_function(X2test)
print "score 3 vs ALL", cls.score(X2test, y_test_temp)
pred3 = cls.predict(X2test)


# class 4 vs ALL
print "CLASS 4 VS ALL"
y_train_temp = copy.deepcopy(y2train)
for i in range(len(y2train)):
    if y_train_temp[i] == 4.0:
        y_train_temp[i] = 1.0
    else:
        y_train_temp[i] = -1.0
y_test_temp = copy.deepcopy(y2test)
for i in range(len(y2train)):
    if y_test_temp[i] == 4.0:
        y_test_temp[i] = 1.0
    else:
        y_test_temp[i] = -1.0

parameters = {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y_train_temp)


print clf.best_params_
cls = SVC(kernel='rbf', C=10.0, gamma=0.001)
cls.fit(X2train, y_train_temp)  # Train on X_train, y_train
# Margin (decision) values of classifier
margins4 = cls.decision_function(X2test)
print "score 4 vs ALL", cls.score(X2test, y_test_temp)
pred4 = cls.predict(X2test)


# Costruisco i due matricioni
print margins0
print margins1
int = np.column_stack((margins0, margins1))

int2 = np.column_stack((int, margins2))
int3 = np.column_stack((int2, margins3))
int4 = np.column_stack((int3, margins4))
binMargins = int4
print "Matricione margini\n"
print binMargins


y_res = np.empty(len(y2train))
y_res.fill(0)

for i in range(len(y2train)):
    # maxY = binY[i][0]
    maxLabel = 0
    # maxMarg = abs(binMargins[i][0])
    maxMarg = binMargins[i][0]
    for j in range(1, 5):
        # currentY = binY[i][j]
        # currentMarg = abs(binMargins[i][j])
        currentMarg = binMargins[i][j]
        # if currentY == 1 and currentY == maxY and currentMarg > maxMarg:
        if currentMarg > maxMarg:
            # maxY = currentY
            maxLabel = j
            maxMarg = currentMarg
    y_res[i] = maxLabel


count = 0
for i in range(len(y2train)):
    if y_res[i] != y2test[i]:
        count = count + 1
#    print "predict %d, true %d" % (y_res[i], y2test[i])
print "Misclassification: %d/%d" % (count, len(y2train))



count0 = list(y_res).count(0.0)
print "count0: ", count0
count1 = list(y_res).count(1.0)
print "count1: ", count1
count2 = list(y_res).count(2.0)
print "count2: ", count2
count3 = list(y_res).count(3.0)
print "count3: ", count3
count4 = list(y_res).count(4.0)
print "count4: ", count4


from sklearn.multiclass import OneVsRestClassifier
svm_clf = SVC(kernel='rbf')
ovr = OneVsRestClassifier(svm_clf)
ovr.fit(X2train, y2train)
print ovr.score(X2test, y2test)
y_pred = ovr.predict(X2test)
count = 0
for i in range(len(y2train)):
    if y_pred[i] != y2test[i]:
        count = count + 1
print "Misclassification: %d/%d" % (count, len(y2train))




from sklearn import grid_search
parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 1.0, 2.0, 10.0, 20.0, 100.0], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001]}
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X2train, y2train)

print clf.best_params_
# print clf.best_score_
# best_c = clf.best_params_.C
# best_gamma = clf.best_params.gamma
# best_kernel = clf.best_params_.kernel
svr = SVC(kernel='rbf', C=10.0, gamma=0.001)
svr.fit(X2train, y2train)
print svr.score(X2test, y2test)
pre = svr.predict(X2test)
num_div = 0
for i in range(len(pre)):
    if pre[i] != y2test[i]:
        num_div = num_div + 1

print "Numeri diversi %d / %d" % (num_div, len(y2test))

svm_clf = SVC(C=10, kernel='rbf', gamma=0.001)
ovr = OneVsRestClassifier(svm_clf)
ovr.fit(X2train, y2train)
pre = ovr.predict(X2test)
print ovr.score(X2test, y2test)
num_div = 0
for i in range(len(pre)):
    if pre[i] != y2test[i]:
        num_div = num_div + 1

print "Numeri diversi %d / %d" % (num_div, len(y2test))

svm_clf = SVC(C=2.0, kernel='rbf', gamma=0.01)
ovr = OneVsRestClassifier(svm_clf)
ovr.fit(X2train, y2train)
pre = ovr.predict(X2test)
print ovr.score(X2test, y2test)
num_div = 0
for i in range(len(pre)):
    if pre[i] != y2test[i]:
        num_div = num_div + 1

print "Numeri diversi %d / %d" % (num_div, len(y2test))
