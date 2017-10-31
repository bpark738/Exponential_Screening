# Briton Park, Amey Mahagan
# STAT 251 

# RigTy2010_Exp.py contains the experiments in Rigolett and Tysbakov (2010) that we ran 
# to test our implementation of the Metropolis approximation algorithm of the Expoential 
# Screening estimator and compare it to ordinary least squares (OLS), lasso 
# regression, and stepwise regression

import MH
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

def generate_data(n, M, beta,s):
	# generate_data outputs generated data using the normal distribution 
	# with dimension n by M for X and n by 1 for Y

	# Inputs:
		# n: number of rows
		# M: number of columns
		# beta: true underlying linear relation (M by 1)
		# s: sparsity parameter
	# Outputs:
		# X: generated data with dimensions n by M
		# Y: generated labels with dimensions n by 1
	
	M = int(M)
	n = int(n)

	w = np.random.normal(0,1, (n, 1))
	X = np.random.normal(0, 1, (n, M))
	Y = (X.dot(beta) + s*w).reshape(n)

	return pd.DataFrame(X), Y

def generate_rad(n, M, beta, s):
	# generate_rad outputs generated data using the rademacher distribution 
	# with dimension n by M for X and n by 1 for Y

	# Inputs:
		# n: number of rows
		# M: number of columns
		# beta: true underlying linear relation coefficients (M by 1)
		# s: sparsity parameter 
	# Outputs:
		# X: generated data with dimensions n by M
		# Y: generated labels with dimensions n by 1

	M = int(M)
	n = int(n)

	w = np.random.normal(0,1, (n,1))
	rad = [-1,1]

	X = (np.random.randint(2, size=(n, M)) - 0.5)*2
	Y = (X.dot(beta) + s*w).reshape(n)

	return pd.DataFrame(X), Y

def setUpInput():
	# setUpInput returns a dataframe with different values for n (number of rows),
	# p (number of columns, s (sparisty parameter), mu (mean parameter for normal distribution,
    # sig_norm (standard deviation parameter for normal distribution), sigma (ES parameter))
	
	df = pd.DataFrame(np.zeros((2, 10)))
	df.columns = ['n', 'p', 'sigma', 'sparsity', 'mu_norm', 'sig_norm', 'ols', 'stepwise', 'lasso', 'ES']
	
	n_p = [(100,200, 10), (200,500, 20)]

	count = 0

	for n, p,s in (n_p):
		df['n'][count] = n
		df['p'][count] = p
		df['sparsity'][count] = s
		df['mu_norm'][count] = 0
		df['sig_norm'][count] = 1
		df['sigma'][count] = np.sqrt(s/9)
		count +=1
	return df

def compute_ES(thetas, T0, T):
	# compute_ES computes the ES estimator using the last
	# T estimates

	# Inputs:
	#	thetas: a list of T0 + T ES estimates
	#   T0: starting index of estimator to include in calculation
	#   T: number of estimators to include in calculation

	return np.sum(thetas[T0+1:T0 +T],0)/T

def run():
	data = setUpInput()

	for i in range(data.shape[0]):
		T0 = 3000
		T = 7000
		print i
		s = int(data['sparsity'][i])
		sig = data['sigma'][i]

		# Obtain true beta coefficients

		beta = np.zeros(int(data['p'][i])).reshape((int(data['p'][i]), 1))
		beta[:s] = 1

		# Generate data

		X_train, Y_train = generate_rad(data['n'][i], data['p'][i], beta, data['sigma'][i])

		X_test, Y_test = generate_rad(data['n'][i], data['p'][i], beta, data['sigma'][i])

		# Apply ES
		fdict = pd.DataFrame(np.eye(int(data['p'][i])))
		thetas = MH.MH_estimate(X_train, fdict, Y_train, sig)
		theta = MH.compute_ES(thetas, T0, T)

		data['ES'][i] = MH.ES_MSE_Y(Y_test, theta, X_test)

		# Apply OLS
		lr = linear_model.LinearRegression()
		lr.fit(X_train, Y_train)
		data['ols'][i] = np.sum((lr.predict(X_test) - Y_test)**2) / X_test.shape[0]

		# Apply Lasso
		l = linear_model.LassoCV()
		l.fit(X_train, Y_train)

		data['lasso'][i] = np.sum((l.predict(X_test).reshape(X_test.shape[0]) - X_test.dot(beta).as_matrix().reshape(X_test.shape[0])  )**2) / X_test.shape[0]

		# Apply stepwise
		f = feature_selection.f_regression(X_train, Y_train)[1]
		f_ind = np.argsort(f)

		f_mse = []
		for j in range(1,int(data['p'][i])):
			lr = linear_model.LinearRegression()
			lr.fit(X_train.loc[:,f_ind[0:j]], Y_train)
			f_mse.append(np.sum((lr.predict(X_test.loc[:, f_ind[0:j]]) - Y_test)**2) / X_test.shape[0])

		data['stepwise'][i] = min(f_mse)

	data.to_csv("data.csv")

run()