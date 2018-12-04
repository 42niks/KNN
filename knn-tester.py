import numpy as np
import gc
from knn import knn 
from multiprocessing import Pool
from timeit import default_timer as time

def print_metrics(knn_object, path):
	f = open(path+'/'+str(knn_object.K)+'.txt', 'w')
	f.write('Confusion Matrix:\n')
	f.write(str(knn_object.confusion_matrix))
	f.write('\nRecall:\n')
	f.write(str(knn_object.recall))
	f.write('\nMean Recall:\n')
	f.write(str(knn_object.meanrecall))
	f.write('\nPrecision:\n')
	f.write(str(knn_object.precision))
	f.write('\nMean Precision\n')
	f.write(str(knn_object.meanprecision))
	f.write('\nFmeasure:\n')
	f.write(str(knn_object.fmeasure))
	f.write('\nAccuracy:\n')
	f.write(str(knn_object.accuracy))
	f.close()

def main(K, path):
	start = time()
	ultron = knn(K, 3)
	print(K,'loading training data')
	ultron.load_train_from_folder(path+'/Train')
	print(K,'loading testing data')
	ultron.load_test_from_folder(path+'/Test')
	print(K,'testing ultron...')
	ultron.calculate_metrics()
	print(K, 'printing to output')
	print_metrics(ultron, path)
	return 

if __name__ == '__main__':
	K = [2, 4, 8, 16, 32]
	path = 'Group04'
	l = [(k, path) for k in K]
	p = Pool(processes=5)
	result = p.starmap(main, l)
