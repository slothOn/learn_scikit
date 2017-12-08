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
        self.clf = RandomForestClassifier()
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
                if index == 0 or i == index:
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

    def parse3(self):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        for i in range(1, 13):
            with open(self.dirpath + str(i), 'r') as f:
                lines = f.readlines()
                #random.shuffle(lines)
                train_size = 10
                #train_size = int(0.9 * len(lines))
                for j in range(train_size):
                    train_features.append(parse_features(lines[j]))
                    train_labels.append(i)
                for j in range(train_size, len(lines)):
                    test_features.append(parse_features(lines[j]))
                    test_labels.append(i)
        return train_features, train_labels, test_features, test_labels

    def parse4(self, index):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        for i in range(1, 13):
            with open(self.dirpath + str(i), 'r') as f:
                lines = f.readlines()
                #random.shuffle(lines)
                if i == index:
                    train_size = 50
                else:
                    train_size = int(0.9 * len(lines))
                for j in range(train_size):
                    train_features.append(parse_features(lines[j]))
                    train_labels.append(i)
                for j in range(train_size, len(lines)):
                    test_features.append(parse_features(lines[j]))
                    test_labels.append(i)
        return train_features, train_labels, test_features, test_labels


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

    def test_limited(self):
        train_features, train_labels, test_features, test_labels = self.parse3()
        self.clf.fit(train_features, train_labels)
        pred_labels = []
        wrong = 0
        pred_labels = self.clf.predict(test_features)
        for predict, label in zip(pred_labels, test_labels):
            if predict != label:
                wrong += 1
        '''
        for feature, label in zip(test_features, test_labels):
            predict = self.clf.predict(feature)
            pred_labels.append(predict)
            if label != predict:
                wrong += 1
        '''
        allsize = len(test_features)
        score = float(allsize - wrong) / float(allsize)
        print('accuracy score:' + str(score))
        my_confustion_matrix(test_labels, pred_labels, 'test_limited')


    def test_biased(self, index):
        train_features, train_labels, test_features, test_labels = self.parse4(index)
        self.clf.fit(train_features, train_labels)
        pred_labels = []
        wrong = 0
        pred_labels = self.clf.predict(test_features)
        for predict, label in zip(pred_labels, test_labels):
            if predict != label:
                wrong += 1
        allsize = len(test_features)
        score = float(allsize - wrong) / float(allsize)
        print('accuracy score:' + str(score))
        my_confustion_matrix(test_labels, pred_labels, 'test_biased')

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

rf = limited_rf('/Users/zxc/Desktop/18697/team/activities_lessfeatures/')
#rf.test(1)
#rf.test_cvreport(1)

#rf.test_limited()
rf.test_biased(12)
