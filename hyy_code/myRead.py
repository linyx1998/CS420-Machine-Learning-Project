import numpy as np

npz_path = "./new_dataset/sketchrnn_"
animal = ["bear", "camel", "cat", "cow", "crocodile", "dog", "elephant", "flamingo", "giraffe", "hedgehog","horse", "kangaroo",\
		"lion", "monkey", "owl","panda","penguin", "pig", "raccoon", "rhinoceros", "sheep", "squirrel", "tiger","whale", "zebra"]

#数据集形状：一个数据集2500个sketch，一个sketch有三行

class read_data():
	
	def __init__(self, isone=False):
		self.test = np.array([], dtype=np.float32)
		self.train = np.array([], dtype=np.float32)
		self.valid = np.array([], dtype=np.float32)
		self.test_label = []
		self.train_label = []
		self.valid_label = []
		count = 0
		for a in animal:
			data = np.load(npz_path+a+"_png.npz",allow_pickle=True, encoding='latin1')
			if count == 0:
				self.test = data['test'].astype(np.float32)
				self.train = data['train'].astype(np.float32)
				self.valid = data['valid'].astype(np.float32)
				self.test_label = np.zeros(shape=(2500),dtype=int)
				self.train_label = np.zeros(shape=(70000),dtype=int)
				self.valid_label = np.zeros(shape=(2500),dtype=int)
			else:
				self.test = np.concatenate((self.test, data['test']))
				self.train = np.concatenate((self.train, data['train']))
				self.valid = np.concatenate((self.valid, data['valid']))
				self.test_label = np.concatenate((self.test_label, np.full((2500), count)))
				self.train_label = np.concatenate((self.train_label, np.full((70000), count)))
				self.valid_label = np.concatenate((self.valid_label, np.full((2500), count)))
			count += 1
		
		self.train = self.train.reshape((self.train.shape[0], 1,28,28))
		self.test = self.test.reshape((self.test.shape[0], 1,28,28))
		self.valid = self.valid.reshape((self.valid.shape[0], 1,28,28))

		#内存不够了，注释掉，尝试使用单通道vgg
		# self.train = np.concatenate((self.train, self.train, self.train), axis=1)
		# self.test = np.concatenate((self.test, self.test, self.test), axis=1)
		# self.valid = np.concatenate((self.valid, self.valid, self.valid), axis=1)
		# print("trainset shape: ",self.train.shape)

		#reshape成1维向量
		if(isone==True):
			self.train = self.train.reshape((self.train.shape[0], 28*28))
			self.test = self.test.reshape((self.test.shape[0], 28*28))
			self.valid = self.valid.reshape((self.valid.shape[0], 28*28))
			
			print("trainset shape: ",self.train.shape)
			print("testset shape: ",self.test.shape)
			print("validset shape: ",self.valid.shape)
		

		#shuffle trainset
		# print(self.train_label)
		# state = np.random.get_state()
		# np.random.shuffle(self.train)
		# np.random.set_state(state)
		# np.random.shuffle(self.train_label)

		# print(self.train_label)

	def get_data(self):
		return self.train, self.test, self.valid, self.train_label, self.test_label, self.valid_label



