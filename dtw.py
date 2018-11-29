import numpy as np
import math
from scipy.spatial.distance import euclidean as ec_dist

"""dtw.py: Implements the Dynamic time warping algo"""
__author__		= "Nikhil Tumkur Ramesh"
__copyright__	= "Copyright 2018, IIT Mandi" 

# def ec_dist(X, Y):
# 	print(X, Y)
# 	print(X-Y)
# 	return math.sqrt(np.sum((X-Y)**2))

def roundoff(num):
	# roundoff_to_places = 2
	return (int(num*(100)))/(100)

def dist(X, Y):
	d = ec_dist(X, Y)
	# print(roundoff(d))
	return roundoff(d)

def initialize(mat, X, Y):
	'''
	n: stores the length of the sequence X
	'''
	n = X.shape[0]
	mat[0, 0] = dist(X[0], Y[0])
	if Y.shape[0] > 1:
		mat[0, 1] = mat[0,0] + dist(X[0], Y[1])
	for i in range(1, n):
		mat[i,0] = mat[i-1,0] + dist(X[i], Y[0])
		

def dtw(X, Y):
	'''
	Note: I want the longer one to be X
	mat: rows->X, columns->Y
		dist(X[0], Y[0])		dist(X[0], Y[1])
	'''

	# make the larger one as X so that I don't have to bother about edge cases.
	if X.shape[0] < Y.shape[0]:
		X, Y = Y, X

	# N, M stores the length of sequence X, Y
	N = X.shape[0]
	M = Y.shape[0]

	# initialize the matrix that holds my distances
	mat=np.zeros((N, min(2, M)), dtype=float)
	initialize(mat, X, Y)
	
	# If legth of sequence Y is just 1, then I'm done.
	if M == 1:
		return mat[n-1]

	for m in range(1, M):
		mat[0,1] = mat[0,0] + dist(X[0], Y[m])
		for n in range(1, N):
			# print('value of n', n)
			mat[n, 1] = dist(X[n], Y[m]) + min(mat[n, 0],
												  mat[n-1, 0],
												  mat[n-1, 1])
		# transfer all this data into the old column so that space required is
		# reduced.
		# print(mat)
		# input()
		mat[:,0] = mat[:,1]
		
	# print('onelast time:')
	# print(mat)
	# print(mat[N,0])
	# return time normalalized sequence
	return roundoff(mat[N-1,0]/(N*M))


class DTW():

	X = np.array([])
	Y = np.array([])
	N = 0
	M = 0
	dtw_distance = 0.0

	def __init__(self, X=None, Y=None):
		self.X = X
		self.Y = Y
		if X!=None:
			self.N = X.shape[0]
		else:
			self.N = 0
		if Y!=None:
			self.M = Y.shape[0]
		else:
			self.M = 0
		self.dtw = 0.0

	def setX(self, X):
		self.X = X
		self.N = X.shape[0]

	def setY(self, Y):
		self.Y = Y
		self.M = Y.shape[0]

	def perform_dtw(self, X=None, Y=None):
		if X!=None:
			self.setX(self, X)
		if Y!=None:
			self.setY(self, Y)
		self.dtw_distance = dtw(self.X, self.Y)
		return self.dtw_distance
	
	def get_dtw_dist(self):
		return self.dtw_distance