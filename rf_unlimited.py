__author__ = 'zxc'

from sklearn.ensemble import RandomForestClassifier
from util import *
from sklearn.model_selection import cross_val_predict

class rf_8_2:
    def __init__(self, dirpath = '/Users/zxc/Desktop/18697/team/users/'):
        self.dirpath = dirpath

    def parse(self, index):
        all_features = []
        all_labels = []
        with open(self.dirpath + 'feature_' + str(index), 'r') as f1, open(self.dirpath + 'label_' + str(index), 'r') as f2:
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
        pred_labels = cross_val_predict(clf, X, Y, cv=10)
        my_confustion_matrix(true_label=Y, pred_label=pred_labels, matrix_name='Random Forest large dataset')
        # crossValScore = cross_val_score(clf, allFeatures, allLabels, cv=10)
        # crossVal.append(crossValScore.mean())
        # print("cross val score: " + str(crossValScore.mean()))

def test_unlimited_rf():
    #rf1 = rf_8_2()
    #rf1.test()
    rf1 = rf_8_2('/Users/zxc/Desktop/18697/team/users_limited/')
    rf1.test()

test_unlimited_rf()