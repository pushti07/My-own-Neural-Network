# Creating a neuron with 4 inputs
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias)

print(output)





# Creating a layer of Neurons
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):

    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    
    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

print(layer_outputs)





# Creating a neuron using Numpy
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# convert lists into numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)

outputs = np.dot(weights_array, inputs_array) + bias

print(output)




# Layer of Neurons using Numpy
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)

layer_outputs = np.dot(weights_array, inputs_array) + biases_array

print(layer_outputs)




# We need to take transpose of weight matrix
inputs = [[1.0, 2.0, 3.0, 2.5], 
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)

outputs = np.dot(inputs_array, weights_array.T) + biases_array

print(outputs)





# 2 layers
inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)
weights2_array = np.array(weights2)
biases2_array = np.array(biases2)

# Calculate outputs for the first layer
layer1_output = np.dot(inputs_array, weights_array.T) + biases_array

# Calcualate the output of the second layer
layer2_output = np.dot(layer1_output, weights2_array.T) + biases2_array

print(layer2_output)





# Create Dataset
pip install nnfs

from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

import matplotlib.pyplot as plt

X, y = spiral_data(samples = 100, classes = 3)
plt.scatter(X[ : , 0], X[ : , 1])
plt.show()

plt.scatter(X[ : , 0], X[ : , 1], c = y, cmap = 'brg')
plt.show()





# Implementing Dense Layer class
nnfs.init()

class Layer_Dense:

    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward Pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2,3)

dense1.forward(X)

print(dense1.output[:5])





# Activation fn : Relu
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)
print(output)

class Activation_ReLu:
    # Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Create Dataset
X, y = spiral_data(samples = 100, classes = 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLu()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Takes in output from previous layer
activation1.forward(dense1.output)

print(activation1.output[:5])

