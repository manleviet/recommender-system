from numpy import *
from scipy.optimize import fmin_ncg
from functools import wraps

def hypothesis(features, weights):
	'''Predict ratings given features and weights'''
	return dot(features, transpose(weights))

def optimize(features, weights, ratings, regularization = 0, callback = None):
	assert features.shape[1] == weights.shape[1]

	num_movies, num_features = features.shape
	num_users = weights.shape[0]

	# Stack verically so fmin_ncg gets a single array
	x0 = vstack((features, weights)).flatten()

	# Helper function to unpack the array into two matrices
	def unpack(x):
		x=x.reshape((num_movies + num_users, num_features))
		features = x[:num_movies, :]
		weights = x[num_movies:, :]
		return (features, weights)

	# Wrap the callback with a function to unpack the params
	if callback:
		@wraps(callback)
		def new_callback(x, *args):
			f,w = unpack(x)
			callback(f, w, *args)
	else:
		new_callback = None

	# Helper functions to calculate cost and gradient
	def f(x, *args):
		features,weights = unpack(x)
		return cost(features, weights, ratings)

	def fprime(x, *args):
		features,weights = unpack(x)
		unrated = ratings == 0
		h = hypothesis(features, weights)

		# Ignore unrated
		h[unrated] = 0

		g0 = dot((h-ratings), weights) + regularization * features
		g1 = dot(transpose(h-ratings), features) + regularization * weights
		return array(vstack((g0, g1))).flatten()

	x = fmin_ncg(f, x0, fprime, callback=new_callback)

	return unpack(x)

def cost(features, weights, ratings, regularization=0):
	'''Calculate the cost of the current feature set and weights, given the ratings.
	Ratings is a matrix num_movies x num_users.
	Features is a matrix num_movies x num_features
	Weights is a matrix num_users x num_features'''
	unrated = ratings == 0 # zero indicates unrated; these movies should not contribute to the cost
	h = hypothesis(features, weights)
	h[unrated] = 0
	return 0.5*sum(power(h-ratings, 2)) + regularization / 2 * (sum(power(features, 2)) + sum(power(weights, 2)))

def grad_features(features, weights, ratings, regularization=0):
	'''Calculate the gradients of the cost function with respect to the features.'''
	unrated = ratings == 0
	h = hypothesis(features, weights)

	# Ignore unrated
	h[unrated] = 0

	return dot((h-ratings), weights) + regularization * features

def grad_weights(features, weights, ratings, regularization=0):
	'''Calculate the gradients of the cost function with respect to the weights.'''
	unrated = ratings == 0
	h = hypothesis(features, weights)

	# Ignore unrated
	h[unrated] = 0

	return dot(transpose(h-ratings), features) + regularization * weights
