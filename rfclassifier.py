__author__ = 'zxc'

import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from util import *

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import random

class limited_rf:
    def __init__(self, dirpath='/Users/zxc/Desktop/18697/team/activities/'):
        self.dirpath = dirpath
        self.clf = RandomForestClassifier(n_estimators=10)
        #self.clf = GaussianNB()
        #self.clf = linear_model.LogisticRegression()

    def count(self):
        for i in range(1, 13):
            with open(self.dirpath + str(i), 'r') as f:
                lines = f.readlines()
                print(len(lines))

    def parse(self, index):
        all_features = []
        all_labels = []
        for i in range(1, 13):
            with open(self.dirpath + str(i), 'r') as f:
                lines = f.readlines()
                if i == index:
                    lines = random.sample(lines, 20)
                for line in lines:
                    all_features.append(parse_features(line))
                    all_labels.append(i)
        return all_features, all_labels

    def parse2(self):
        all_features = []
        all_labels = []
        for i in range(1, 13):
            with open(self.dirpath + str(i), 'r') as f:
                lines = f.readlines()
                lines = random.sample(lines, 20)
                for line in lines:
                    all_features.append(parse_features(line))
                    all_labels.append(i)
        return all_features, all_labels

    def test(self, index):
        features, labels = self.parse(index)
        scores = cross_val_score(self.clf, features, labels, cv=10, scoring='accuracy')
        print(scores)
        print(scores.mean())

    def test2(self):
        features, labels = self.parse2()
        scores = cross_val_score(self.clf, features, labels, cv=10, scoring='accuracy')
        print(scores)
        print(scores.mean())

    def test_cvreport(self, index):
        if index == 0:
            features, labels = self.parse2()
        else:
            features, labels = self.parse(index)
        pred_labels = cross_val_predict(self.clf, features, labels, cv=10)
        my_confustion_matrix(true_label=labels, pred_label=pred_labels)

def test_limited_unbiased(rf):
    # test all limited
    rf.test2()
    rf.test_cvreport(0)

def test_limited_biased(rf):
    #rf.count()
    '''
    # test bias on 12
    rf3.test(12)
    rf3.test_cvreport(12)
    # test all limited
    rf3.test2()
    rf3.test_cvreport(0)
    '''
    path = 'Users/zxc/Desktop/18697/team/activities_lessfeatures/'
    for i in range(1, 13):
        rf.test(i)
        rf.test_cvreport(i)

#rf = limited_rf('/Users/zxc/Desktop/18697/team/activities_lessfeatures/')
#rf = limited_rf()
#test_limited_biased(rf)

rf = limited_rf()
rf.count()