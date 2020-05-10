from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
info = clf.fit(X, y)
print(info)

import pickle
s = pickle.dumps(clf)
clf2=pickle.loads(s)
info = clf2.predict(X[0:1])
print(info)
print(y[0])
