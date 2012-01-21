from numpy import *
from scipy.optimize import fmin_ncg
from functools import wraps

class Optimizer(object):
	'''Optimize the features/weights to minimise Optimizer.cost'''

	@staticmethod
	def optimize(features, weights, ratings, regularization = 0, callback=None):
		'''Returns the optimum features and weights'''
		return Optimizer(features, weights, ratings, regularization, callback).run()

	@staticmethod
	def cost(features, weights, ratings, regularization=0):
		'''Calculate the cost of the feature set and weights, given the ratings.
		Ratings is a matrix num_movies x num_users.
		Features is a matrix num_movies x num_features
		Weights is a matrix num_users x num_features'''
		rated = ratings != 0
		h = dot(features, transpose(weights))[rated]
		return 0.5 * sum(power(h-ratings[rated], 2)) + regularization / 2 * (sum(power(features, 2)) + sum(power(weights, 2)))

	@staticmethod
	def hypothesis(features, weights):
		'''Predict ratings given features and weights'''
		return dot(features, transpose(weights))

	def __init__(self, features, weights, ratings, regularization = 0, callback = None):
		'''Initialize an optimizer object. It is not neccessary to call this directly, as the static optimize method returns the optimal parameters.'''
		assert features.shape[1] == weights.shape[1]
	
		self.num_movies, self.num_features = features.shape
		self.num_users = weights.shape[0]

		# Stack vertically so fmin_ncg gets a single array
		self.x = vstack((features, weights)).flatten()
		assert len(self.x.shape) == 1

		self.rated = ratings != 0
		self.ratings = ratings[self.rated]
		self.regularization = regularization

		# Wrap the callback with a function to unpack the params
		if callback:
			@wraps(callback)
			def new_callback(x, *args):
				f,w = self.unpack(x)
				callback(f, w, *args)
		else:
			new_callback = None

		self.callback = new_callback

	def run(self):
		'''Run the optimization and return the optimal parameters.'''
		self.x = fmin_ncg(self.f, self.x, self.fprime, callback=self.callback)
		return self.params()

	def unpack(self, x):
		'''Helper function to unpack the feature array into two matrices.'''
		boundary = self.num_movies * self.num_features
		features = x[:boundary].reshape((self.num_movies, self.num_features))
		weights = x[boundary:].reshape((self.num_users, self.num_features))
		return (features, weights)

	def params(self):
		'''Return the optimized parameters, i.e. features and weights.'''
		return self.unpack(self.x)

	# Helper functions to calculate cost and gradient
	def f(self, x, *args):
		'''Calculate the cost'''
		features, weights = self.unpack(x)
		h = Optimizer.hypothesis(features, weights)[self.rated]

		return 0.5*sum(power(h-self.ratings, 2)) + self.regularization / 2 * (sum(power(features, 2)) + sum(power(weights, 2)))


	def fprime(self, x, *args):
		'''Calculate the gradients of the cost function with respect to each feature and weight.'''
		features, weights = self.unpack(x)
		h = Optimizer.hypothesis(features, weights)

		diffs = zeros(h.shape)
		diffs[self.rated] = h[self.rated] - self.ratings
		g0 = dot(diffs, weights) + self.regularization * features
		g1 = dot(transpose(diffs), features) + self.regularization * weights
		return array(vstack((g0, g1))).flatten()
