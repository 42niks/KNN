import numpy as np
from kmeans import kmeans
import gc
from multiprocessing import Pool
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def do_kmeans(file, K):
	print("Launching K-Means for:", file, K)
	f=open(file, 'r')
	lines = f.readlines()
	f.close()
	f=None
	w=[]
	for point in lines:
		w.append(np.fromstring(point, dtype=float, sep=' '))
	points=np.array(w)
	w=None
	point=None
	lines=None
	gc.collect()

	print('loaded points:')
	# print(points)
	start = timer()
	friday = kmeans(K)
	friday.fit(points, n_clusters=K, display_progress=True, debug_mode=True)
	print(file, K,": Fit took %f seconds"%(timer()-start))
	print(file, K, ": writing to output")
	means = friday.cluster_means()
	list_of_points = friday.list_of_points
	colour = ['green', 'red', 'yellow', 'blue', 'pink', 'brown']
	for k, l in enumerate(list_of_points):
		plt.plot(points[l,0], points[l,1], 'o', color=colour[k%len(colour)], markersize=5)
	for idx, mean in enumerate(means):
		plt.plot(mean[0], mean[1], marker='$'+str(idx)+'$', c='black', markersize=12)
	plt.show()
	output=open(file[:-4]+"K"+str(K)+".txt", 'w')
	for mean in means:
		for number in mean:
			output.write(str(number)+' ')
		output.write('\n')
	output.close()
	print("Done with:", file, K)
	return 1

if __name__ == "__main__":
	list_of_files = ["TrainingData\\all.txt"]
	clusters = [16, 32]
	l = [(file, cluster) for cluster in clusters for file in list_of_files]
	p=Pool(processes=2)
	results=p.starmap(do_kmeans, l)