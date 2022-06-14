import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from read import read_data
import time

from TSNE import draw, draw_PCA, draw_original

def PCA_SVM():
	r = read_data(num = 500)
	train, test, train_label, test_label = r.get_data()
	#复制一份训练集，后面直接对原始数据降维
	Var = [0.9]
	original_train = train
	original_test = test

	#不降维的结果
	# svc = svm.SVC(C=1.0, kernel='rbf')
	# svc.fit(train,train_label)
	# y_val_pred = svc.predict(test)        
	# num_correct = np.sum(y_val_pred == test_label)
	# num_val = test.shape[0]
	# accuracy = float(num_correct) / num_val
	# print("without PCA: ", accuracy)
	for each_var in Var:
		start_time = time.time()  
		#复原
		train = original_train
		test = original_test

		###找到能够保留95%方差的n_components
		pca_1=PCA(n_components=each_var, copy = False)  #copy = False指直接对数据集降维
														#调用PCA，会自动均值归一化
		X_reduce = pca_1.fit_transform(train)
		dim = X_reduce.shape[1]
		
		###利用上面找到的n_components，降维
		pca=PCA(n_components=dim, copy = False)
		X_reduce_fianl =pca.fit_transform(train)
		test_reduce_finel = pca.fit_transform(test)
		
		### 利用支持向量机训练
		# svc = svm.SVC(C=1.0, kernel='rbf')
		# svc.fit(X_reduce_fianl,train_label)
		# # accuracy = svc.score(test_reduce_finel,test_label)
		# y_val_pred = svc.predict(test_reduce_finel)        
		# num_correct = np.sum(y_val_pred == test_label)
		# num_val = test.shape[0]
		# accuracy = float(num_correct) / num_val
		
		# end_time = time.time()
		# Total_time = end_time - start_time
		# print("var : ", each_var,"dimension: ", X_reduce_fianl.shape,"total time: ", Total_time ,"accuracy : ",accuracy)

		#TSNE可视化
		# draw_original(original_train, train_label, 'original.png', 'data distribution of original data')
		draw_PCA(X_reduce_fianl, train_label, 'PCA_data.png', 'data distribution of PCA reduced data')

PCA_SVM()