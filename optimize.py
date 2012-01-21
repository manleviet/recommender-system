'''Optimize movie features and user weights.'''
# Redefining built in sum function is A-OK.
# pylint: disable-msg=W0622
from numpy import dot, power, transpose, vstack, array, sum, zeros
from scipy.optimize import fmin_cg
from functools import wraps

class Optimizer(object):
	'''Optimizes the movie features and user weights to minimise Optimizer.cost.
	Optimizer objects operate on the following matrices:

		features (num_movies x num_features): Learned features for the movies
		weights (num_users x num_features):   Learned weights for the users
		rated (num_movies x num_users):       Boolean values showing which movies
		                                      users have rated
		ratings (num_movies x num_users):     The ratings (1-10) for each movie the
		                                      users have rated.
		
		                                      N.B. This is converted to a vector
		                                      where each element corresponds to a
		                                      True value in a flattened version of
		                                      the `rated` matrix (assuming a C-style
		                                      array layout).

		                                      The original matrix can be restored
		                                      as follows:
		                                          a = zeros(o.rated.shape)
		                                          a[o.rated] = o.ratings
	'''

	@staticmethod
	def optimize(features, weights, ratings, regularization = 0, callback=None,
			**kwargs):
		'''Returns the optimal features and weights for a ratings matrix.
		features, weights, ratings should all be NumPy arrays.
		Any additional keyword arguments (e.g. maxiter) are passed through to the
		underlying optimization function in SciPy.'''
		return Optimizer(features, weights, ratings, regularization, callback,
				**kwargs).run()

	@staticmethod
	def cost(features, weights, ratings, regularization=0):
		'''Calculate the cost of the feature set and weights, given the ratings'''
		rated = ratings != 0
		h = dot(features, transpose(weights))[rated]
		return 0.5 * sum(power(h-ratings[rated], 2)) + \
				0.5 * regularization * (sum(power(features, 2)) + sum(power(weights, 2)))

	@staticmethod
	def hypothesis(features, weights):
		'''Predict ratings given features and weights'''
		return dot(features, transpose(weights))

	def __init__(self, features, weights, ratings, regularization = 0,
			callback = None, **kwargs):
		'''Initialize an optimizer object. It should not be neccessary to call this
		directly, as the static optimize method returns the optimal parameters.'''
		assert features.shape[1] == weights.shape[1]
	
		self.num_movies, self.num_features = features.shape
		self.num_users = weights.shape[0]

		# Stack vertically so fmin_cg gets a single array
		self.x = vstack((features, weights)).flatten()

		self.rated = ratings != 0
		self.ratings = ratings[self.rated]
		self.regularization = regularization

		# Wrap the callback with a function to unpack the params
		if callback:
			@wraps(callback)
			def new_callback(x, *args):
				'''wrapper'''
				f, w = self.unpack(x)
				callback(f, w, *args)
		else:
			new_callback = None

		self.callback = new_callback
		self.options = kwargs

	def run(self):
		'''Run the optimization and return the optimal parameters.'''
		self.x = fmin_cg(self.f, self.x, self.fprime, callback=self.callback,
				**self.options)
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
	def f(self, x):
		'''Calculate the cost'''
		features, weights = self.unpack(x)
		h = Optimizer.hypothesis(features, weights)[self.rated]

		return 0.5 * (sum(power(h-self.ratings, 2)) + \
				self.regularization * (sum(power(features, 2)) + sum(power(weights, 2))))

	def fprime(self, x):
		'''Calculate the gradients of the cost function with respect to each feature
		and weight.'''
		features, weights = self.unpack(x)
		h = Optimizer.hypothesis(features, weights)

		diffs = zeros(h.shape)
		diffs[self.rated] = h[self.rated] - self.ratings
		g0 = dot(diffs, weights) + self.regularization * features
		g1 = dot(transpose(diffs), features) + self.regularization * weights
		return array(vstack((g0, g1))).flatten()
