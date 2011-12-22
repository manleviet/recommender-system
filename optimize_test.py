import optimize
from dataset import dataset,params
from numpy.random import rand
from numpy import *

epsilon = 0.01

class TestOptimize:
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
		cost = optimize.cost(self.features, self.weights, self.ratings)
		assert abs(cost - 22.22) < epsilon

class TestGradientNumerically:
	def setup(self):
		# Initial guesses
		self.features = matrix(rand(4, 3))
		self.weights = matrix(rand(5,3))

		# Used to generate ratings
		true_features = matrix(rand(4, 3))
		true_weights = matrix(rand(5,3))

		self.ratings = true_features * true_weights.T

		# Remove some ratings
		self.ratings[rand(4,5) > 0.5] = 0

	def test_grad_weights(self):
		numeric = numerical_grad_weights(self.features, self.weights, self.ratings)
		analytical = optimize.grad_weights(self.features, self.weights, self.ratings)

		assert all(abs(numeric - analytical) < epsilon)

	def test_grad_features(self):
		numeric = numerical_grad_features(self.features, self.weights, self.ratings)
		analytical = optimize.grad_features(self.features, self.weights, self.ratings)

		assert all(abs(numeric - analytical) < epsilon)

def numerical_grad_weights(features, weights, ratings, regularization=0):
	'''Compute the gradient of the cost_function with respect to the eights'''
	e = 1e-4
	perturb = zeros(weights.shape)
	grad = zeros(weights.shape)

	# Create flattened views over the data
	flat_grad = grad.ravel()
	flat_perturb = perturb.ravel()

	for i in range(weights.size):
		flat_perturb[i] = e

		cost1 = optimize.cost(features, weights - perturb, ratings, regularization)
		cost2 = optimize.cost(features, weights + perturb, ratings, regularization)

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

		cost1 = optimize.cost(features - perturb, weights, ratings, regularization)
		cost2 = optimize.cost(features + perturb, weights, ratings, regularization)

		flat_perturb[i] = 0
		flat_grad[i] = (cost2-cost1) / (2*e)

	return grad
