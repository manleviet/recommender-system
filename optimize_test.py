import optimize

from dataset import dataset,params
from numpy.random import rand
from numpy import *
from nose.tools import timed
from nose.plugins import prof

epsilon = 0.01

# Nosetests unit tests for the optimize module
class TestOptimize(object):
	def setup(self):
		features = params['X']
		weights = params['Theta']
		ratings = dataset['Y']
		rated = dataset['R']
		num_users = 4
		num_movies = 5
		num_features = 3
		self.features = features[:num_movies, :num_features]
		self.weights = weights[:num_users, :num_features]
		self.ratings = ratings[:num_movies, :num_users];

	def test_initial_cost(self):
		cost = optimize.Optimizer.cost(self.features, self.weights, self.ratings)
		assert abs(cost - 22.22) < epsilon

	def test_initial_cost_r(self):
		cost = optimize.Optimizer.cost(self.features, self.weights, self.ratings, regularization=1.5)
		assert abs(cost - 31.34) < epsilon

	def test_cost_reduced(self):
		cost_before = optimize.Optimizer.cost(self.features, self.weights, self.ratings, regularization=1.5)
		features, weights = optimize.Optimizer.optimize(self.features, self.weights, self.ratings, regularization=1.5)
		cost_after = optimize.Optimizer.cost(features, weights, self.ratings, regularization=1.5)
		assert cost_after < cost_before

class TestTime(object):
	def setup(self):
		self.features = params['X']
		self.weights = params['Theta']
		self.ratings = dataset['Y']
		self.rated = dataset['R']
		self.optimizer = optimize.Optimizer(self.features,self.weights,self.ratings,1.5)
		#num_users = 100
		#num_movies = 140
		#num_features = 3
		#self.features = self.features[:num_movies, :num_features]
		#self.weights = self.weights[:num_users, :num_features]
		#self.ratings = self.ratings[:num_movies, :num_users];

	@timed(0.47)
	def test_cost(self):
		self.optimizer.f(self.optimizer.x)
		self.optimizer.fprime(self.optimizer.x)

#	@timed(20)
#	def test_cost_reduced(self):
#		calls=[0]
#		def foo(f,w):
#			calls[0] += 1
#			print '%d: %g' % (calls[0], optimize.Optimizer.cost(f,w,self.ratings, 1.5))
#		features, weights = optimize.Optimizer.optimize(self.features, self.weights, self.ratings, regularization=1.5,callback=foo)

class TestGradientNumerically(object):
	def setup(self):
		# Initial guesses
		self.features = rand(4, 3)
		self.weights = rand(5,3)

		# Used to generate ratings
		true_features = rand(4, 3)
		true_weights = rand(5,3)

		self.ratings = dot(true_features, transpose(true_weights))

		# Remove some ratings
		self.ratings[rand(4,5) > 0.5] = 0

	def test_grad(self):
		optimizer = optimize.Optimizer(self.features, self.weights, self.ratings)
		numeric_f = numerical_grad_features(self.features, self.weights, self.ratings)
		numeric_w = numerical_grad_weights(self.features, self.weights, self.ratings)
		numeric = vstack((numeric_f, numeric_w)).flatten()
		assert numeric.shape == (27,)
		assert optimizer.x.shape == (27,)
		analytical = optimizer.fprime(optimizer.x)
		assert analytical.shape == (27,)

		assert all(abs(numeric - analytical) < epsilon)

	def test_grad_r(self):
		optimizer = optimize.Optimizer(self.features, self.weights, self.ratings, regularization=1.5)
		numeric_f = numerical_grad_features(self.features, self.weights, self.ratings, regularization=1.5)
		numeric_w = numerical_grad_weights(self.features, self.weights, self.ratings, regularization=1.5)
		numeric = vstack((numeric_f, numeric_w)).flatten()
		analytical = optimizer.fprime(optimizer.x)

		assert all(abs(numeric - analytical) < epsilon)

def numerical_grad_weights(features, weights, ratings, regularization=0):
	'''Compute the gradient of the cost_function with respect to the weights'''
	e = 1e-4
	perturb = zeros(weights.shape)
	grad = zeros(weights.shape)

	# Create flattened views over the data
	flat_grad = grad.ravel()
	flat_perturb = perturb.ravel()

	for i in range(weights.size):
		flat_perturb[i] = e

		cost1 = optimize.Optimizer.cost(features, weights - perturb, ratings, regularization)
		cost2 = optimize.Optimizer.cost(features, weights + perturb, ratings, regularization)

		flat_perturb[i] = 0
		flat_grad[i] = (cost2-cost1) / (2*e)

	return grad

def numerical_grad_features(features, weights, ratings, regularization=0):
	'''Compute the gradient of the cost_function with respect to the features'''
	e = 1e-4
	perturb = zeros(features.shape)
	grad = zeros(features.shape)

	# Create flattened views over the data
	flat_grad = grad.ravel()
	flat_perturb = perturb.ravel()

	for i in range(features.size):
		flat_perturb[i] = e

		cost1 = optimize.Optimizer.cost(features - perturb, weights, ratings, regularization)
		cost2 = optimize.Optimizer.cost(features + perturb, weights, ratings, regularization)

		flat_perturb[i] = 0
		flat_grad[i] = (cost2-cost1) / (2*e)

	return grad

# When the script is run directly, plot the learning curves instead
if __name__ == '__main__':
	import matplotlib.pyplot as plt

	class LearningCurve(object):
		def __init__(self, features, weights, ratings, xfeatures, xratings):
			self.costs = []
			self.xcosts = []
			self.ratings = ratings
			self.xratings = xratings
			self.xfeatures = xfeatures
			optimize.Optimizer.optimize(features, weights, ratings, callback=self.update)

		def update(self, features, weights, *args):
			self.costs.append(optimize.Optimizer.cost(features, weights, self.ratings))
			self.xcosts.append(optimize.Optimizer.cost(self.xfeatures, weights, self.xratings))
			print self.costs[-1]

		def plot(self):
			plt.subplot('211')
			plt.plot(self.costs)
			plt.subplot('212')
			plt.plot(self.xcosts)

	features = params['X']
	weights = params['Theta']
	ratings = dataset['Y']
	rated = dataset['R']
	num_users = 200
	num_movies = 500
	num_features = 3
	features = features[:num_movies, :num_features]
	weights = weights[:num_users, :num_features]
	ratings = ratings[:num_movies, :num_users];

	# Rest of the movies make up cross validation set
	xfeatures = params['X'][num_movies:, :num_features]
	xratings = dataset['Y'][num_movies:, :num_users];

	LearningCurve(features, weights, ratings, xfeatures, xratings).plot()

	plt.show()
