from read import read_data
from sklearn import neighbors
import numpy as np
from sklearn.decomposition import PCA
from metric_learn import LMNN, NCA, LFDA, MLKR
import time

def KNN():
	N = [10000]
	Metric = [ 'euclidean', 'chebyshev','manhattan']
	
	for each_n in N:
		for each_m in Metric:
			r = read_data(num = each_n)
			train, test, train_label, test_label = r.get_data()

			start_time = time.time()
			knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, metric= each_m)
			knn_classifier.fit(train, train_label)
			y_val_pred = knn_classifier.predict(test)        
			num_correct = np.sum(y_val_pred == test_label)
			num_val = test.shape[0]
			accuracy = float(num_correct) / num_val
			end_time = time.time()

			print("train_size: ", each_n ,"metric: ", each_m,"total time: ", end_time-start_time, "accuracy: ", accuracy)

def PCA_KNN():
	r = read_data(num = 10000)
	train, test, train_label, test_label = r.get_data()
	N = [0.99, 0.9]
	
	#PCA降维
	Var = [0.9]
	original_train = train
	original_test = test

	#不降维的结果

	# start_time = time.time()
	# knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, metric= 'euclidean')
	# knn_classifier.fit(train, train_label)
	# y_val_pred = knn_classifier.predict(test)        
	# num_correct = np.sum(y_val_pred == test_label)
	# num_val = test.shape[0]
	# accuracy = float(num_correct) / num_val
	# end_time = time.time()

	# print("dimension: ", train.shape[0] ,"metric: ", 'euclidean',"total time: ", end_time-start_time, "accuracy: ", accuracy)
	
	#降维的结果
	
	for each_n in N:

		#复原
		train = original_train
		test = original_test

		###找到能够保留95%方差的n_components
		pca_1=PCA(n_components=each_n, copy = False)  #copy = False指直接对数据集降维
														#调用PCA，会自动均值归一化
		X_reduce = pca_1.fit_transform(train)
		dim = X_reduce.shape[1]
		
		###利用上面找到的n_components，降维
		pca=PCA(n_components=dim, copy = False)
		X_reduce_fianl =pca.fit_transform(train)
		test_reduce_finel = pca.fit_transform(test)
		
		start_time = time.time()
		knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, metric= 'euclidean')
		knn_classifier.fit(X_reduce_fianl, train_label)
		y_val_pred = knn_classifier.predict(test_reduce_finel)        
		num_correct = np.sum(y_val_pred == test_label)
		num_val = test.shape[0]
		accuracy = float(num_correct) / num_val
		end_time = time.time()

		print("dimension: ", each_n ,"metric: ", 'euclidean',"total time: ", end_time-start_time, "accuracy: ", accuracy)

def Learning_KNN():
	r = read_data(num=100)
	train, test, train_label, test_label = r.get_data()

	print("现在是LFDA方法")
	time_start = time.time()
	lmnn = LFDA()
	lmnn.fit(train,train_label)
	X_train_new = lmnn.transform(train)
	X_test_new = lmnn.transform(test)
	time_end=time.time()
	print('time cost for metric learning is: ',time_end-time_start,'s')

	#KNN
	start_time = time.time()
	knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, metric= 'euclidean')
	knn_classifier.fit(X_train_new, train_label)
	y_val_pred = knn_classifier.predict(X_test_new)        
	num_correct = np.sum(y_val_pred == test_label)
	num_val = test.shape[0]
	accuracy = float(num_correct) / num_val
	end_time = time.time()

	print("dimension: ", X_train_new.shape ,"learning: ", LMNN,"total time: ", end_time-start_time, "accuracy: ", accuracy)


Learning_KNN()