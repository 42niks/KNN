import numpy as np
from dtw import DTW
import os

def bubble_sort(K, distances, c_labels):
	n = len(distances)
	for i in range(K):
		for j in range(n-1, i, -1):
			if distances[j]<distances[j-1]:
				distances[[j,j-1]] = distances[[j-1,j]]
				c_labels[[j,j-1]] = c_labels[[j-1,j]]

class knn:
	N = 0 # number of classes
	K = 0 # window size
	X = [] # training data set
	c_labels = np.array([], dtype=int) # classlabels for corresponding data
	Y = [] # testing data set
	actual_labels = np.array([], dtype=int) # actual class labels
	assigned_classlabel = np.array([], dtype=int)
	recall = []
	precision = []
	fmeasure = []
	accuracy = 0.0

	def __init__(self, K, N):
		self.K = K
		self.N = N
		self.confusion_matrix = np.ones((N, N), dtype=int)

	def classifySeq(self, input_seq):
		# print('input seq:', input_seq)
		self.class_labels = np.copy(self.c_labels)
		self.distances = np.zeros((len(self.X)), dtype=float)
		dtw = DTW()
		dtw.setY(input_seq)
		for i, seq in enumerate(self.X):
			# print('\tcomparing with', i)
			dtw.setX(seq)
			self.distances[i] = dtw.perform_dtw()
		bubble_sort(self.K, self.distances, self.class_labels)
		self.bins = np.bincount(self.class_labels[:self.K])
		self.assigned_classlabel = self.bins.argmax()
		# print('assigned class label: ', self.assigned_classlabel)
		return self.assigned_classlabel

	def load_train_from_folder(self, path):
		self.X, self.c_labels =  self.input_from_folder(path)
		# print('loaded training data')
		# print('type of X', type(self.X))
		# print('length of X', len(self.X))

	def load_test_from_folder(self, path):
		self.Y, self.actual_labels = self.input_from_folder(path)

	############################################################################
	###################### Performance Testing #################################
	############################################################################

	def calculate_metrics(self):
		self.assigned_classlabels =  self.assign_classlabels(self.Y, 
			self.actual_labels)
		self.confusion_matrix = self.calculate_confusion_matrix(
			self.actual_labels,
			self.assigned_classlabels)
		sumrows = np.sum(self.confusion_matrix, axis=1)
		sumcols = np.sum(self.confusion_matrix, axis=0)
		for i in range(self.N):
			self.recall.append(self.confusion_matrix[i][i]/sumrows[i])
			self.precision.append(self.confusion_matrix[i][i]/sumcols[i])
			self.fmeasure.append((2*self.precision[0]*self.recall[0])
				/(self.precision[0]+self.recall[0]))
			self.accuracy+=self.confusion_matrix[i][i]
		self.accuracy = self.accuracy / (sum(sumcols))
		self.meanrecall = np.mean(self.recall)
		self.meanprecision = np.mean(self.precision)
		
	def assign_classlabels(self, Y, actual_labels):
		assigned_classlabels = np.copy(actual_labels)
		for i, y in enumerate(Y):
			print(self.K,'assigning class label to', i)
			assigned_classlabels[i] = self.classifySeq(y)
			# print('given:', assigned_classlabels[i])
		return assigned_classlabels

	def calculate_confusion_matrix(self, actual_labels, assigned_classlabels):
		confusion_matrix = np.zeros((self.N, self.N), dtype=int)
		for i in range(actual_labels.shape[0]):
			confusion_matrix[actual_labels[i], assigned_classlabels[i]]+=1
		return confusion_matrix

	############################################################################
	########################### File I/O Module ################################
	############################################################################

	# reads one particular file
	def read_file(self, data, path, folder, file):
		f = open(path+'/'+folder+'/'+file, 'r')
		lines=f.readlines()

		# adds a numpy matrix to my training data list
		data.append(np.array([[float(x)for x in line.split()]
			for line in lines]))
				
	# reads one particular class folder
	def read_folder(self, data, path, folder):
		files = os.listdir(path+'/'+folder)
		no_of_files=0
		for file in files:
			if not os.path.isfile(path+'/'+folder+'/'+file):
				continue
			self.read_file(data, path, folder, file)
			no_of_files+=1
		return no_of_files

	# reads a directory filled with class folders
	def input_from_folder(self, path):
		data = []
		labels = np.array([], dtype=int)
		folders = os.listdir(path)
		class_label = 0
		for folder in folders:
			if not os.path.isdir(path+'/'+folder):
				continue
			no_of_files=self.read_folder(data, path, folder)
			labels = np.append(labels, np.array([class_label 
				for i in range(no_of_files)], dtype=int))
			class_label+=1
		# print('length of data = ', len(data))
		return data, labels

