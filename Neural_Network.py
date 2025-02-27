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





# Activation fn : Softmax
class Activation_softmax:

    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))

        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        self.output = probabilities


# Create Dataset
X, y = spiral_data(samples = 100, classes = 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLu()

# Create second Dense layer with 3 input features
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# it takes the output of first dense layer here
activation1.forward(dense1.output)

# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# it takes the output of second dense layer here
activation2.forward(dense2.output)

print(activation2.output[:5])




# Calculate Network error with loss:

# Cross Entropy loss building blocks

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_output[range(len(softmax_output)), class_targets])

neg_log = -np.log(softmax_output[range(len(softmax_output)), class_targets])

print("Loss : ", neg_log)

average_loss = np.mean(neg_log)

print("Average Loss : ", average_loss)




# If Data is OneHot Encoded, How to extract Relevant Predictions ?
y_true_check = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])

y_pred_clipped_check = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

A = y_true_check * y_pred_clipped_check
B = np.sum(A, axis = 1)
C = -np.log(B)

print("Loss : ", C)
print("Average Loss : ", np.mean(C))





# Implementing the Loss class:
class Loss:
    # given model output and ground truth values
    def calculate(self, output, y):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss
    
# Implementing Categorical Cross Entropy:
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if(len(y_true.shape) == 1):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        elif (len(y_true.shape) == 2):
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods
    
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

loss_function = Loss_CategoricalCrossEntropy()

loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)





# Introducing Accuracy:
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

predictions = np.argmax(softmax_outputs, axis = 1)

if(len(class_targets.shape) == 2):
    class_targets = np.argmax(class_targets, axis = 1)

accuracy = np.mean(predictions == class_targets)

print('accuracy : ', accuracy)




# Need for optimization : 
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()
X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()



# Optimization Strategy  1 : Randomly select weights and biases (Doesn't work !)
X, y = vertical_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
 # Generate a new set of weights for iteration
 dense1.weights = 0.05 * np.random.randn(2, 3)
 dense1.biases = 0.05 * np.random.randn(1, 3)
 dense2.weights = 0.05 * np.random.randn(3, 3)
 dense2.biases = 0.05 * np.random.randn(1, 3)

 # Perform a forward pass of the training data through this layer
 dense1.forward(X)

 activation1.forward(dense1.output)

 dense2.forward(activation1.output)

 activation2.forward(dense2.output)

 # Perform a forward pass through activation function
 # it takes the output of second dense layer here and returns loss
 loss = loss_function.calculate(activation2.output, y)

 # Calculate accuracy from output of activation2 and targets
 # calculate values along first axis
 predictions = np.argmax(activation2.output, axis=1)
 accuracy = np.mean(predictions == y)
 
 # If loss is smaller - print and save weights and biases aside
 if loss < lowest_loss:
   print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', accuracy)
   best_dense1_weights = dense1.weights.copy()
   best_dense1_biases = dense1.biases.copy()
   best_dense2_weights = dense2.weights.copy()
   best_dense2_biases = dense2.biases.copy()
   lowest_loss = loss





# Strategy 2 : Randomly adjust weights and biases (works !)
# Create dataset
X, y = vertical_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
 # Update weights with some small random values
 dense1.weights += 0.05 * np.random.randn(2, 3)
 dense1.biases += 0.05 * np.random.randn(1, 3)
 dense2.weights += 0.05 * np.random.randn(3, 3)
 dense2.biases += 0.05 * np.random.randn(1, 3)

 # Perform a forward pass of our training data through this layer
 dense1.forward(X)

 activation1.forward(dense1.output)

 dense2.forward(activation1.output)

 activation2.forward(dense2.output)

 # Perform a forward pass through activation function
 # it takes the output of second dense layer here and returns loss
 loss = loss_function.calculate(activation2.output, y)

 # Calculate accuracy from output of activation2 and targets
 # calculate values along first axis
 predictions = np.argmax(activation2.output, axis=1)
 accuracy = np.mean(predictions == y)

 # If loss is smaller - print and save weights and biases aside
 if loss < lowest_loss:
  print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', accuracy)
  best_dense1_weights = dense1.weights.copy()
  best_dense1_biases = dense1.biases.copy()
  best_dense2_weights = dense2.weights.copy()
  best_dense2_biases = dense2.biases.copy()
  lowest_loss = loss
  
 # Revert weights and biases
 else:
  dense1.weights = best_dense1_weights.copy()
  dense1.biases = best_dense1_biases.copy()
  dense2.weights = best_dense2_weights.copy()
  dense2.biases = best_dense2_biases.copy()





# Strategy 2 : For spiral Dataset
# Create dataset
X, y = spiral_data(samples=100, classes=3)# Create model

dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()


# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
 # Update weights with some small random values
 dense1.weights += 0.05 * np.random.randn(2, 3)
 dense1.biases += 0.05 * np.random.randn(1, 3)
 dense2.weights += 0.05 * np.random.randn(3, 3)
 dense2.biases += 0.05 * np.random.randn(1, 3)

 # Perform a forward pass of our training data through this layer
 dense1.forward(X)

 activation1.forward(dense1.output)

 dense2.forward(activation1.output)

 activation2.forward(dense2.output)

 # Perform a forward pass through activation function
 # it takes the output of second dense layer here and returns loss
 loss = loss_function.calculate(activation2.output, y)

 # Calculate accuracy from output of activation2 and targets
 # calculate values along first axis
 predictions = np.argmax(activation2.output, axis=1)
 accuracy = np.mean(predictions == y)

 # If loss is smaller - print and save weights and biases aside
 if loss < lowest_loss:
  print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', accuracy)
  best_dense1_weights = dense1.weights.copy()
  best_dense1_biases = dense1.biases.copy()
  best_dense2_weights = dense2.weights.copy()
  best_dense2_biases = dense2.biases.copy()
  lowest_loss = loss
  
 # Revert weights and biases
 else:
  dense1.weights = best_dense1_weights.copy()
  dense1.biases = best_dense1_biases.copy()
  dense2.weights = best_dense2_weights.copy()
  dense2.biases = best_dense2_biases.copy()
