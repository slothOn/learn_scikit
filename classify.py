__author__ = 'zxc'

import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
import numpy as np

def parse(path, numSubject):
	dict = {}
	for i in range(1, numSubject + 1):
		dict[i] = [];
		file = path + str(i) + ".csv"
		with open(file) as f:
			reader = csv.reader(f)
			for row in reader:
				dict[i].append(row)
	return dict

# Generate train and test data from parsing the file
path = "course_proj/REALDISP/"
mhealthSubject = 10
dict = parse(path, mhealthSubject)
allVectors = []
for i in range(1, mhealthSubject + 1):
	allVectors.extend(dict[i])
	random.shuffle(dict[i])
random.shuffle(allVectors)
allFeatures = []
allLabels = []
for i in range(len(allVectors)):
	allFeatures.append(allVectors[i][:-1])
	allLabels.append(allVectors[i][-1])
userTrainFeatures = []
userTestFeatures = []
userTrainLabels = []
userTestLabels = []
for user in dict:
	userFeatures = []
	userLabels = []
	for i in range(len(dict[user])):
		userFeatures.append(map(np.float, dict[user][i][:-1]))
		userLabels.append(dict[user][i][-1])
	length = len(userFeatures)
	userTrainFeatures.extend(userFeatures[:int(length * 0.7)])
	userTestFeatures.extend(userFeatures[int(length * 0.7):])
	userTrainLabels.extend(userLabels[:int(length * 0.7)])
	userTestLabels.extend(userLabels[int(length * 0.7):])

import matplotlib.pyplot as plt
#plt.xlabel('Number of depth')
#plt.ylabel('Performance')

trainErrors = list()
testErrors = list()
crossVal = list()

# clf = DecisionTreeClassifier()
# clf.fit(userTrainFeatures, userTrainLabels)
# score = clf.score(userTestFeatures, userTestLabels)
# print("score: " + str(score))

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(userTrainFeatures, userTrainLabels)
# score = gnb.score(userTestFeatures, userTestLabels)
# print("score: " + str(score))

logreg = linear_model.LogisticRegression()
logreg.fit(userTrainFeatures, userTrainLabels)
score = logreg.score(userTestFeatures, userTestLabels)
print("score: " + str(score))



# for n in [10, 20, 30]:
# 	clf = RandomForestClassifier(n_estimators=10, max_depth=n)
# 	clf.fit(userTrainFeatures, userTrainLabels)
# 	predict = clf.predict([userTestFeatures[0]])
# 	print("Built random forest.")
# 	trainScore = clf.score(userTrainFeatures, userTrainLabels)
# 	trainErrors.append(trainScore)
# 	print("Train score: " + str(trainScore))
# 	testScore = clf.score(userTestFeatures, userTestLabels)
# 	testErrors.append(testScore)
# 	print("Test score: " + str(testScore))
# 	crossValScore = cross_val_score(clf, allFeatures, allLabels, cv=10)
# 	crossVal.append(crossValScore.mean())
# 	print("cross val score: " + str(crossValScore.mean()))
# plt.plot([10, 20, 30], trainErrors, label='Train')
# plt.plot([10, 20, 30], testErrors, label='Test')
# plt.plot([10, 20, 30], crossVal, label='10 Fold Cross Validation')
# plt.legend()
# plt.show()