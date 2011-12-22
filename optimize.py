from numpy import *

def hypothesis(features, weights):
	'''Predict ratings given features and weights'''
	features = matrix(features)
	weights = matrix(weights)
	return features * weights.T

def optimize(features, weights, ratings, regularization = 0, callback = None):
	assert features.shape == weight.shape

	# Stack in 3rd dimension so fmin_ncg gets a single array
	x0 = dstack(features, weights)

	# Helper functions to calculate cost and gradient
	def f(x, *args):
		features = matrix(x[:,:,0])
		weights = matrix(x[:,:,1])
		return self.cost(features, weights, ratings)

	def fprime(x, *args):
		features = matrix(x[:,:,0])
		weights = matrix(x[:,:,1])
		g0 = self.grad_features(features, weights, ratings)
		g2 = self.grad_weights(features, weights, ratings)
		return dstack(g0, g2)

	x = fmin_ncg(f, x0, fprime, callback=callback)

	# Unwrap feature and weight matrices
	features = params[:,:,0]
	weights = params[:,:,1]

	return (features, weights)

def cost(features, weights, ratings, regularization=0):
	'''Calculate the cost of the current feature set and weights, given the ratings.
	Ratings is a matrix num_movies x num_users.
	Features is a matrix num_movies x num_features
	Weights is a matrix num_users x num_features'''
	unrated = ratings == 0 # zero indicates unrated; these movies should not contribute to the cost
	h = hypothesis(features, weights)
	h[unrated] = 0
	return 0.5*sum(power(h-ratings, 2))

def grad_features(features, weights, ratings, regularization=0):
	'''Calculate the gradients of the cost function with respect to the features.'''
	unrated = ratings == 0
	h = hypothesis(features, weights)

	# Ignore unrated
	h[unrated] = 0

	return (h-ratings) * weights

def grad_weights(features, weights, ratings, regularization=0):
	'''Calculate the gradients of the cost function with respect to the weights.'''
	unrated = ratings == 0
	h = hypothesis(features, weights)

	# Ignore unrated
	h[unrated] = 0

	return (h-ratings).T * features
