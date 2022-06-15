import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

def mySVM(table,target,parametrs):
    x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=0.10)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return [x_test,y_pred,y_test]
