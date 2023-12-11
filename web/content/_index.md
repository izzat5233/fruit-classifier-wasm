+++
title = 'Home'
date = 2023-01-01T08:00:00-07:00
draft = false
+++

# Fruism | Fruit Classifier in Web Assembly

The FCNN is a lightweight, flexible neural network library written in C++, designed for the specific task of classifying
different types of fruits. This library forms the core part of the larger Fruit Classifier project.

## Features:

- *Customizable Neural Network*: Create neural networks with varying structures, including different numbers of layers,
  neurons per layer, and activation functions.
- *Multiple Activation Functions*: Includes several built-in activation functions like step, sign, linear, ReLU,
  sigmoid,
  and tanh, each with its corresponding derivative for backpropagation.
- *Efficient Computation*: Optimized for performance, utilizes WebAssembly to allow fast c++ near-native speed.
- *Gradient Descent Training*: Utilizes gradient descent algorithm for network training.

## Components

- **```nn```**: Main namespace containing all components.

- **```Neuron```**: Represents a single neuron, encapsulating weights and bias.
  Provides functionality for computing neuron's output.

- **Layer Types**:
    - **```Layer```**: Base class for layers in the neural network.
    - **```HiddenLayer```**: Represents a hidden layer in the network.
    - **```OutputLayer```**: Special layer type using Softmax activation.

- **```Network```**:Represents the entire neural network, a collection of layers.
  Implements forward and backward propagation methods for network training.

- **Activation Functions**: Defined in the ```act``` namespace with built-in functions for use in network layers.
  Includes a special softmax function for output layers.

- **Loss Functions**: Defined in the ```loss``` namespace with built-in functions for use in network layers.

- **Factory Functions**:
  Located in the ```make``` namespace, these functions allow for the creation of Neurons, Layers, and Networks with
  specific configurations.

## Usage

- To create a neural network:
    - Define Network Structure: Determine the number of layers and neurons in each layer.
    - Select Activation Functions: Choose activation functions for each hidden layer.
    - Create Network: Use the ```nn::make::network``` function to create a network instance.
    - Train the Network: Provide training data and use ```nn::Network::train``` method to train the network.

- Example

```c++
#include <network.h>

int main() {
    // Create a neural network with specific dimensions and activation functions
    // The following network has 2 inputs and a hidden layers with 2 neurons,
    // and output layer with 2 neurons. With an initial learning rate of 0.01.
    // First hidden layer uses ReLU activation function, second one uses Sigmoid.
    nn::Network network = nn::make::network({2, 2, 2}, {nn::act::relu, nn::act::sigmoid}, 0.01);

    // Define the data. Typically, a vector of decimals vectors.
    nn::vpvd_t data = {       // This is an XOR dataset
            {{0, 0}, {1, 0}}, // for 0 xor 0 we expect a 0
            {{0, 1}, {0, 1}}, // for 0 xor 1 we expect a 1
            {{1, 0}, {0, 1}}, // for 1 xor 0 we expect a 1
            {{1, 1}, {1, 0}}  // for 1 xor 1 we expect a 0
    };
    
    // Train the network with your data...
    // Keep training until it reaches 100 epochs
    // or a maximum sum square error 0.1
    for (std::size_t i = 0; i < 100; ++i) {
        auto error = network.train(data, nn::loss::sse);
        if (error <= 0.1) { break; }
    }
    

    // Now use it to predict outputs...
    // vd_t is a vector of double values.
    // example result {0.129248 0.870752}
    nn::vd_t output = network.predict({0, 1});
    
    return 0;
}
```
