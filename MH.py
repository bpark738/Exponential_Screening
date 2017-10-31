# Briton Park, Amey Mahajan
# STAT 251 

# MH.py contains the implementation for the Metropolis Approximation 
# of the Exponential Screening (ES) estimator

import numpy as np
import pandas as pd
from random import randint
import random
import math
from sklearn import datasets
from numpy.linalg import matrix_rank
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import feature_selection

def compute_OLS(Y, X_small):
	# compute_OLS returns the OLS estimator

	# Inputs:
	#	Y: n by 1 labels
	#	X: n by p data matrix

	# Output:
	#	theta: OLS estimator

	theta = np.linalg.lstsq(X_small, Y)[0]
	return theta

def sparse_OLS(Y, design, p):
	# sparse_OLS returns the OLS estimator for non-zero positions

	# Inputs:
	#	Y: n by 1 labels
	#	design: n by M data matrix
	#   p: sparsity pattern (M by 1 vector of 0's and 1's representing presence or absence of a feature)

	# Output:
	#	theta: sparse estimator ()

	if sum(p) == 0:
		return p
	index= [item[0] for item in enumerate(p) if item[1] == 1]

	X_small = design.ix[:, index]
	theta_small = compute_OLS(Y, X_small)

	theta = np.zeros(len(p))
	j = 0
	for i in index:
		theta[i] = theta_small[j]
		j += 1

	return theta

def compute_diff(theta, design, Y):
	# compute_diff returns the squared difference between the labels and the predictions

	# Inputs: 
	#  Y: n by 1 labels
	#  design: design matrix n by M

	# Output: squared difference between labels and predictions
	return np.power(Y - design.dot(theta),2)

def compute_prior_ratio(p, q, design):
	# compute_prior_ratio returns the ratio of the priors for sparse vectors p and q
	# via RigTy (2010) Figure 1

	# Inputs
	# p: sparsity pattern 1 (M by 1 vector of 0's and 1's representing presence or absence of a feature)
	# q: sparsity pattern 2 (M by 1 vector of 0's and 1's representing presence or absence of a feature)
	# design: n by M matrix 

	# Output: ratio of priors for sparsity patterns p and q

	w = sum(q) - sum(p)
	ps = sum(p)
	qs = sum(q)
	M = design.shape[1]
	ratio = math.pow((1.0 + (1.0 * w / ps)), qs) * math.pow((ps / (2.0 * math.exp(1) * M)), w)

	return ratio

def ES_MSE(theta_r, theta, design):
	# ES_MSE function computes the mean squared of the predictions using
	# the ES estimator without noise

	# Inputs:
	# theta_r: true theta coefficients (M by 1)
	# theta: ES estimator coefficients (M by 1)
	# design matrix: n by M matrix

	# Output: MSE of ES predictions

	theta = theta.reshape((len(theta), 1))
	return np.sum(((design.dot(theta) -design.dot(theta_r))**2)/design.shape[0])

def ES_MSE_Y(Y, theta, design):
	# ES_MSE_Y function computes the mean squared of the predictions using
	# the true labels (with noise)

	# Inputs:
	# Y: true labels (n by 1)
	# theta: ES estimator coefficients (M by 1)
	# design matrix: n by M matrix

	# Output: MSE of ES predictions

	Y = np.array(Y)
	Y = Y.reshape((len(Y), 1))
	theta = theta.reshape((len(theta), 1))
	return np.sum(((design.dot(theta) -Y)**2))/design.shape[0]

def compute_design(data, fdict):
	# compute_design returns the design matrix of the data

	# Inputs:
	#   data: n by M matrix (original data)
	#   fdict: dictionary of functions (M by M vectors)

	# Output: design matrix computed by taking the dot product
	#   between the functions in fdict and the data matrix

	n = data.shape[0]
	M = data.shape[1]
	design = pd.DataFrame(np.zeros((n, M)))

	for i in range(n):
		for j in range(M):
			design.loc[i,j] = fdict.loc[j, :].dot(data.loc[i, :])

	return design


def MH_estimate(data, fdict, Y, sigma):
	# MH_estimate returns the list of coefficient estimates

	# Inputs:
	#   data: original n by M data matrix
	#   fdict: dictionary of functions (M by M vectors)
	#   Y: true labels (n by 1)
	# 

	# Output:
	#   c: list of computed estimates

	design = compute_design(data, fdict)
	n, M= data.shape
	# initial sparsity pattern
	p = np.zeros(M)
	theta = 0

	c = []

	for i in range(1000):
		q = p.copy()
		rando = randint(0, M-1)
		q[rando]= 1 - p[rando]
		#print "p:", p
		#print "q:", q

		# calculate OLS estimators and take vector norm difference
		theta_p = sparse_OLS(Y, design, p)
		theta_q = sparse_OLS(Y, design, q)
		t1 = compute_diff(theta_p, design, Y)
		t2 = compute_diff(theta_q, design, Y)
		t = np.sum(t1 - t2)

		# finish computing nu_q / nu_p
		add = (sum(p) - sum(q)) / 2.0
		sig = 1 / (4.0 * math.pow(sigma, 2))
		llh = sig*(t) + add

		# calculate prior ratio as suggested in paper
		pr = compute_prior_ratio(p, q, data)

		# compute ratio
		ratio = np.exp(llh) * pr

		# compute sampling probability r(p, q)
		r = min(ratio, 1.0)

		if sum(p) == 0:
			r = 1
		# generate random variable
		rv = random.uniform(0, 1)

		print "r", r
		if(rv <= r):
			print "rv less than r"
			p = q.copy()
			theta = theta_q.copy()
		else:
			print "else condition triggered"
			theta = theta_p.copy()
		print "theta", theta
		c.append(theta)

	return c

def compute_ES(thetas, T0, T):
	# compute_ES computes the ES estimator using the last
	# T estimates

	# Inputs:
	#	thetas: a list of T0 + T ES estimates
	#   T0: starting index of estimator to include in calculation
	#   T: number of estimators to include in calculation

	# Output: ES estimator obtained by averaging using
	#  MH estimates T0+1 to T0+T

	return np.sum(thetas[T0 + 1 : T0 +T], 0) / T