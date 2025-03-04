# My-own-Neural-Network

This repository contains a Python implementation of a neural network from scratch, including the creation of datasets, forward and backward propagation, activation functions, loss functions, and optimization techniques. The project is designed to help you understand the inner workings of neural networks by building one step-by-step.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Overview](#code-overview)
6. [Optimization Techniques](#optimization-techniques)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

This project is an educational implementation of a neural network, designed to help you understand the fundamental concepts of neural networks, including forward and backward propagation, activation functions, loss functions, and optimization techniques. The implementation is done using Python and NumPy, with no reliance on high-level machine learning libraries like TensorFlow or PyTorch.

## Features

- **Dataset Creation**: Generate spiral and vertical datasets for training and testing.
- **Layer Implementation**: Implement dense layers with customizable input and output sizes.
- **Activation Functions**: Implement ReLU and Softmax activation functions.
- **Loss Functions**: Implement Categorical Cross-Entropy loss for classification tasks.
- **Optimization Techniques**: Implement various optimization algorithms including SGD, Momentum, Adagrad, RMSprop, and Adam.
- **Backpropagation**: Implement backpropagation to update weights and biases.
- **Testing**: Validate the model on test datasets to measure accuracy and loss.

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/your-username/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install nnfs numpy matplotlib
```

## Usage

To train the neural network on the spiral dataset, run the following script:

```python
python train.py
```

This will train the model using the Adam optimizer and display the loss and accuracy at regular intervals.

## Code Overview

### Dataset Creation

The dataset is created using the `spiral_data` function from the `nnfs` library.

```python
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
```

### Layer Implementation

The `Layer_Dense` class implements a dense layer with forward and backward propagation.

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
```

### Activation Functions

The `Activation_ReLu` and `Activation_Softmax` classes implement the ReLU and Softmax activation functions, respectively.

```python
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
```

### Loss Functions

The `Loss_CategoricalCrossEntropy` class implements the Categorical Cross-Entropy loss function.

```python
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
```

### Optimization Techniques

The project includes implementations of various optimization algorithms, including SGD, Momentum, Adagrad, RMSprop, and Adam.

```python
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
```

## Testing

To validate the model, a test dataset is created, and the model's performance is evaluated.

```python
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
```

