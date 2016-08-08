__author__ = 'Francesco'

from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
Read raw pixels of 72 images for three classes and concatenate the classes sample in order to have a 216x49152 matrix X
"""
path = 'obj100/'
class1_sample = np.empty((1, 49152), dtype=np.float)
class1_label = []
for filename in os.listdir(path):
    img_data = np.asarray(Image.open(path+filename))
    x = img_data.ravel()
    class1_sample = np.vstack((class1_sample, x))
    #class1_label.append("obj28")
    class1_label.append(100)
class1_sample = np.delete(class1_sample, 0, axis=0)
class1_label = np.array(class1_label).reshape(72, 1)

assert class1_sample.shape == (72, 49152), "The matrix has not the dimensions 50x49152"
assert class1_label.shape == (72, 1), "The vector has not the dimensions 50x1"


path = 'obj10/'
class2_sample = np.empty((1, 49152), dtype=np.float)
class2_label = []
for filename in os.listdir(path):
    img_data = np.asarray(Image.open(path+filename))
    x = img_data.ravel()
    #print x
    class2_sample = np.vstack((class2_sample, x))
    # class2_label.append("obj4")
    class2_label.append(10)
class2_sample = np.delete(class2_sample, 0, axis=0)
class2_label = np.array(class2_label).reshape(72, 1)

assert class2_sample.shape == (72, 49152), "The matrix has not the dimensions 72x49152"
assert class2_label.shape == (72, 1), "The vector has not the dimensions 72x1"


path = 'obj38/'
class3_sample = np.empty((1, 49152), dtype=np.float)
class3_label = []
for filename in os.listdir(path):
    img_data = np.asarray(Image.open(path+filename))
    x = img_data.ravel()
    #print x
    class3_sample = np.vstack((class3_sample, x))
    # class3_label.append("obj10")
    class3_label.append(38)
class3_sample = np.delete(class3_sample, 0, axis=0)
class3_label = np.array(class3_label).reshape(72, 1)

assert class3_sample.shape == (72, 49152), "The matrix has not the dimensions 72x49152"
assert class3_label.shape == (72, 1), "The vector has not the dimensions 72x1"



X = np.concatenate((class1_sample, class2_sample, class3_sample), axis=0)
y = np.concatenate((class1_label, class2_label, class3_label), axis=0).ravel()
assert X.shape == (216, 49152), "The matrix has not the dimensions 216x49152"
assert y.shape == (216,), "The vector is not a 1-d array with 216 labels"


"""
# Plot the samples for class 1, class 2 and class 3

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
        'o', markersize=8, color='blue', alpha=0.5, label='class28')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
        '^', markersize=8, alpha=0.5, color='red', label='class4')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
        '*', markersize=8, alpha=0.5, color='green', label='class10')


#plt.title('Examples for class 1, class 2 and class 3')
ax.legend(loc='upper right')

#plt.show()
plt.savefig('new_all_samples.png')
plt.close()
"""



"""
Standardizing X
From docs: StandardScaler() standardize features by removing the mean and scaling to unit variance. [..]
           Standardization of a dataset is a common requirement for many machine learning estimators:
           they might behave badly if the individual feature do not more or less look like standard normally
           distributed data (e.g. Gaussian with 0 mean and unit variance).
# Warning: StandardScaler assumes floating point values as input, got int32
"""
X_std = StandardScaler().fit_transform(X)


"""
PCA
"""

pca2 = PCA(n_components=2)
X_t2 = pca2.fit_transform(X_std)

print "Number of components: ", pca2.n_components_
print "*", pca2.explained_variance_ratio_
print np.round(pca2.explained_variance_ratio_, decimals=3)*100


# Visualize data for 1st and 2nd Principal Components

"""
import matplotlib.pyplot as plt
plt.plot(X_t2[0:72, 0], X_t2[0:72, 1],
         'o', markersize=7, color='blue', alpha=0.5, label='obj1')
plt.plot(X_t2[72:144, 0], X_t2[72:144, 1],
         '^', markersize=7, color='red', alpha=0.5, label='obj2')
plt.plot(X_t2[144:216, 0], X_t2[144:216, 1],
         '*', markersize=7, color='green', alpha=0.5, label='obj3')

plt.xlabel('First component')
plt.ylabel('Second component')
plt.xlim([-300, 300])
plt.ylim([-200, 200])
plt.legend()
plt.title('PCA results')
# plt.show()
plt.savefig('new_test1.png')
plt.close()
"""

pca4 = PCA(n_components=4)
X_t4 = pca4.fit_transform(X_std)

print "Number of components: ", pca4.n_components_

print "*", pca4.explained_variance_ratio_
print np.round(pca4.explained_variance_ratio_, decimals=3)*100


# Visualize data for 2nd and 3rd Principal Components
"""
import matplotlib.pyplot as plt
plt.plot(X_t4[0:72, 2], X_t4[0:72, 3],
         'o', markersize=7, color='blue', alpha=0.5, label='obj1')
plt.plot(X_t4[72:144, 2], X_t4[72:144, 3],
         '^', markersize=7, color='red', alpha=0.5, label='obj2')
plt.plot(X_t4[144:216, 2], X_t4[144:216, 3],
         '*', markersize=7, color='green', alpha=0.5, label='obj3')

plt.xlabel('Third component')
plt.ylabel('Fourth component')
plt.xlim([-300, 300])
plt.ylim([-200, 200])
plt.legend()
plt.title('PCA results')
#plt.show()
plt.savefig('new_test2.png')
plt.close()
"""

pca11 = PCA(n_components=11)
X_t11 = pca11.fit_transform(X_std)

print "Number of components: ", pca11.n_components_

print "*", pca11.explained_variance_ratio_
print np.round(pca11.explained_variance_ratio_, decimals=3)*100


# Visualize data for 10th and 11th Principal Components
"""
import matplotlib.pyplot as plt
plt.plot(X_t11[0:72, 9], X_t11[0:72, 10],
         'o', markersize=7, color='blue', alpha=0.5, label='obj1')
plt.plot(X_t11[72:144, 9], X_t11[72:144, 10],
         '^', markersize=7, color='red', alpha=0.5, label='obj2')
plt.plot(X_t11[144:216, 9], X_t11[144:216, 10],
         '*', markersize=7, color='green', alpha=0.5, label='obj3')

plt.xlabel('Tenth component')
plt.ylabel('Eleventh component')
plt.xlim([-300, 300])
plt.ylim([-200, 200])
plt.legend()
plt.title('PCA results')
# plt.show()
plt.savefig('new_test3.png')
plt.close()
"""



print "\n## Classification ##\n"

"""
#Splitting, testing and training data (All samples X)
"""
print "> Splitting, testing and training data (all samples X)\n"
# If test_size=None, the value is automatically set to the complement of the train size.
# If train size is also None, test size is set to 0.25.
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
fit1 = gnb.fit(X_train, y_train)
print "Score: ", fit1.score(X_test, y_test)


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    target_names = ['obj28', 'obj4', 'obj10']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)


y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix ALL')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#plt.show()
plt.savefig('Confusion_matrix_all.png')
plt.close()

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))




"""
#Splitting, testing and training for the data projected onto the first and second principal components
"""
print "\n> Splitting, testing and training for the data projected onto 1st and 2nd PC\n"
r1c1 = X_t2[0:72, 0].reshape(72, 1)
r1c2 = X_t2[0:72, 1].reshape(72, 1)
r2c1 = X_t2[72:144, 0].reshape(72, 1)
r2c2 = X_t2[72:144, 1].reshape(72, 1)
r3c1 = X_t2[144:216, 0].reshape(72, 1)
r3c2 = X_t2[144:216, 1].reshape(72, 1)

first = np.hstack((r1c1, r1c2))
second = np.hstack((r2c1, r2c2))
third = np.hstack((r3c1, r3c2))
X2 = np.concatenate((first, second, third), axis=0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.3, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
fit2 = gnb.fit(X_train2, y_train2)
print "Score (1st, 2nd): ", fit2.score(X_test2, y_test2)
