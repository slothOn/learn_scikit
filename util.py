__author__ = 'zxc'

from sklearn.metrics import confusion_matrix
import re
from evaluate_model import *

def myprint(val, file = None):
    if file:
        file.writeline(val)
    else:
        print(val)

def my_confustion_matrix(true_label, pred_label, matrix_name):
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



    tp = conf_matrix[11][11]
    fp = 0
    for i in range(11):
        fp += conf_matrix[i][11]
    tn = 0
    for i in range(11):
        for j in range(11):
            tn += conf_matrix[i][j]
    fn = 192 - tp
    p = float(tp) / (tp + fp)
    r = float(tp) / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('f1 score:' + str(f1))

    plot_cm(conf_matrix, list(range(1, 13)), matrix_name)


def parse_features(line1):
    try:
        features = re.split(',', line1)
        features = list(map(float, features))
    except BaseException:
        print(line1)
    return features