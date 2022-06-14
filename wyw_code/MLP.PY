
from sklearn.neural_network import MLPClassifier
from read import read_data

import datetime

def MLP():
	r = read_data(num=5000)
	train, test, train_label, test_label = r.get_data()
	iter = [5000,10000,20000,40000,80000]
	for i in iter:
		acc = []
		for a in range(3):
			starttime = datetime.datetime.now() 

			clf = MLPClassifier(solver= 'sgd', alpha=1e-3, hidden_layer_sizes=(500,), activation='logistic', max_iter=i)
			clf.fit(train, train_label)
			accuracy = clf.score(test,test_label)

			endtime = datetime.datetime.now()
			Total_time = (endtime - starttime).seconds
			acc.append(accuracy)
		AVG = 0
		for j in range(3):
			AVG += acc[j]
		print("iteration: ", i,"total time: ", Total_time ,"avg accuracy : ",AVG/3, "accuracy: ", acc)

MLP()