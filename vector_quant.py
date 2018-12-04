import numpy as np
from kmeans import kmeans
import gc
from multiprocessing import Pool
import threading as th
from timeit import default_timer as timer
import numpy as np
import os

def vect_quant(input_file_folder, input_file_name, X, means_file, K):
	# print('Launching Thread vect_quant for ', input_file_folder, input_file_name, K)
	f=open(means_file+'K'+str(K)+'.txt', 'r')
	lines = f.readlines()
	f.close()
	f=None
	w=[]
	for point in lines:
		w.append(np.fromstring(point, dtype=float, sep=' '))
	means=np.array(w)
	w=None
	point=None
	lines=None
	gc.collect()

	friday=kmeans(K)
	friday.means = means
	friday.initialize_clusters(X)
	friday.assign_clusters(X)
	means = None
	gc.collect()
	clusters = friday.clusters
	firday = None
	gc.collect()

	output_folder = input_file_folder+'/vq'+str(K)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_file = open(output_folder+'/'+input_file_name[:-4]+'.txt', 'w')
	for c in clusters:
		output_file.write(str(c)+' ')

	# print('Thread Done with:', input_file_folder, input_file_name, K)

def do_vect_quant(folder, means_file, clusters):
	# print("Launching Process do_vect_quant for:", folder, clusters)
	files = os.listdir(folder)
	c = 0
	dirs = 0
	for file in files:
		if not os.path.isfile(folder+'/'+file):
			dirs+=1
			continue
		f=open(folder+'/'+file, 'r')
		lines = f.readlines()
		f.close()
		f=None
		w=[]
		for point in lines:
			w.append(np.fromstring(point, dtype=float, sep=' '))
		X=np.array(w)
		w=None
		point=None
		lines=None
		gc.collect()

		threads = []
		for K in clusters:
			threads.append(th.Thread(target=vect_quant, args=(folder, file, X, means_file, K)))
			threads[-1].start()
		# wait for all threads to complete
		for thread in threads:
			thread.join()
		X = None
		gc.collect()
		c+=1
	# print(folder, "performed operation on", c, "out of", len(files)-dirs, 'files.')
	return 1

if __name__ == '__main__':
	means_file='TrainingData/all'
	clusters = [8,16,32]
	classes = [1, 2, 3]
	fol = ['Test', 'Train']
	folders = ['Group04/'+f+'/class'+str(c) for f in fol for c in classes]
	l=[(folder, means_file, clusters) for folder in folders]
	# print(l)
	p = Pool(processes=4)
	start = timer()
	res = p.starmap(do_vect_quant, l)
	print('task took ', timer()-start)