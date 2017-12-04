__author__ = 'zxc'

import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import random

def my_confustion_matrix(true_label, pred_label):
    labels = list(set(true_label))
    conf_matrix = confusion_matrix(true_label, pred_label, labels=labels)
    print('confusion_matrix(left_labels:true_label, up_labels:pred_labels):')
    print('labels\t', end='')
    for i in range(len(labels)):
        print(str(labels[i]) + '\t', end='')
    print()
    for i in range(len(labels)):
        print(str(labels[i]) + '\t\t', end='')
        for j in range(len(conf_matrix[i])):
            print(str(conf_matrix[i][j]) + '\t', end='')
        print()


def parse_features(line1):
    try:
        features = re.split(',', line1)
        features = list(map(float, features))
    except BaseException:
        print(line1)
    return features


class limited_rf:
    def __init__(self):
        self.dirpath = '/Users/zxc/Desktop/18697/team/activities/'
        self.clf = RandomForestClassifier(n_estimators=10)

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

class rf_8_2:
    def parse(self, index):
        all_features = []
        all_labels = []
        dirpath = '/Users/zxc/Desktop/18697/team/users/'
        with open(dirpath + 'feature_' + str(index), 'r') as f1, open(dirpath + 'label_' + str(index), 'r') as f2:
            line1 = f1.readline().strip()
            line2 = f2.readline().strip()
            while line1 and line2:
                all_features.append(parse_features(line1))
                label = int(line2)
                all_labels.append(label)
                line1 = f1.readline().strip()
                line2 = f2.readline().strip()
        return all_features, all_labels

    def test(self):
        X = []
        Y = []
        for i in range(1, 9):
            x, y = self.parse(i)
            X.extend(x)
            Y.extend(y)
        X2 = []
        Y2 = []
        for i in range(9, 11):
            x, y = self.parse(i)
            X2.extend(x)
            Y2.extend(y)

        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X, Y)
        print('random forest built')
        trainScore = clf.score(X, Y)
        # 	trainErrors.append(trainScore)
        print("Train score: " + str(trainScore))
        testScore = clf.score(X2, Y2)
        # 	testErrors.append(testScore)
        print("Test score: " + str(testScore))
        # crossValScore = cross_val_score(clf, allFeatures, allLabels, cv=10)
        # crossVal.append(crossValScore.mean())
        # print("cross val score: " + str(crossValScore.mean()))

#rf1 = rf_8_2()
#rf1.test()

rf2 = limited_rf()
#rf2.count()

#rf2.test(12)
#rf2.test_cvreport(12)
#rf2.test(6)
rf2.test2()

#rf2.test_cvreport(6)

rf2.test_cvreport(0)