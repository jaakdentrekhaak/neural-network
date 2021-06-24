# Title: Neural Network (brain)
# Author: Jens Putzeys
# Start Date: 2020-09-12 (y-m-d)

import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def matrix_sigmoid(mat):
	result = np.zeros((len(mat), len(mat[0])))
	mfunc = np.vectorize(sigmoid)
	for i in range(len(mat)):
		result[i] = mfunc(mat[i])
	return result


def dsigmoid(x):
	'''
	Derivative of sigmoid. The formula for the derivative of sigmoid(x) = sigmoid(x)*(1-sigmoid(x)), but the input of
	this dsigmoid function already is sigmoid(x).
	'''
	return x * (1 - x)


def matrix_dsigmoid(mat):
	result = np.zeros((len(mat), len(mat[0])))
	mfunc = np.vectorize(dsigmoid)
	for i in range(len(mat)):
		result[i] = mfunc(mat[i])
	return result


class Brain:
	'''
	The 'brain' exists of one input layer, one hidden layer and one output layer.
	'''

	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		# Initializing the weight matrices and biases with random values
		self.weights_ih = np.random.rand(hidden_nodes, input_nodes)  # Weights from input to hidden
		self.weights_ho = np.random.rand(output_nodes, hidden_nodes)  # Weights from hidden to output
		self.bias_h = np.random.rand(hidden_nodes, 1)
		self.bias_o = np.random.rand(output_nodes, 1)

		self.learning_rate = 0.1

	def feed_forward(self, inputs):
		# Transform arguments to numpy matrices
		inputs = [[x] for x in inputs]
		inputs = np.array(inputs)
		hidden = self.weights_ih.dot(inputs)
		hidden += self.bias_h
		hidden = matrix_sigmoid(hidden)  # Activation function
		outputs = self.weights_ho.dot(hidden)
		outputs += self.bias_o
		outputs = matrix_sigmoid(outputs)  # Activation function
		return outputs

	def train(self, inputs, targets):
		# Feed forward the input to calculate the hidden and output
		# We need the hidden values, so we can't just only use self.feed_forward()
		# Transform arguments to numpy matrices
		inputs = [[x] for x in inputs]
		targets = [[x] for x in targets]
		inputs = np.array(inputs)
		targets = np.array(targets)
		hidden = self.weights_ih.dot(inputs)
		hidden += self.bias_h
		hidden = matrix_sigmoid(hidden)  # Activation function
		outputs = self.weights_ho.dot(hidden)
		outputs += self.bias_o
		outputs = matrix_sigmoid(outputs)  # Activation function

		# Calculate the errors
		output_errors = targets - outputs

		# We need to tweak the weights depending on the errors.
		# FORMULA for change in weights: delta_W_ho = learning_rate*Error*(Output*(1-Output)).H_transposed
		# FORMULA for change in bias: delta_b_ho = learning_rate*Error*(Output*(1-Output))
		# In the above formulas, Output*(1-Output) is equal to the derivative of the sigmoid function
		gradient_o = matrix_dsigmoid(outputs)
		a = np.multiply(output_errors, gradient_o)  # Elementwise multiplication
		b = np.multiply(a, self.learning_rate)
		hidden_T = np.transpose(hidden)
		delta_W_ho = b.dot(hidden_T)
		delta_b_ho = b

		gradient_h = matrix_dsigmoid(hidden)
		hidden_errors = np.transpose(self.weights_ho).dot(output_errors)
		a = np.multiply(hidden_errors, gradient_h)
		b = np.multiply(a, self.learning_rate)
		inputs_T = np.transpose(inputs)
		delta_W_ih = b.dot(inputs_T)
		delta_b_ih = b

		# Change the weights and biases
		self.weights_ho += delta_W_ho
		self.bias_o += delta_b_ho
		self.weights_ih += delta_W_ih
		self.bias_h += delta_b_ih
