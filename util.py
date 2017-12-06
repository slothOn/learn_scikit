__author__ = 'zxc'

from sklearn.metrics import confusion_matrix
import re

def myprint(val, file = None):
    if file:
        file.writeline(val)
    else:
        print(val)

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