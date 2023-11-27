# Fruit Classifier Neural Network in WASM (FCNN)

The FCNN is a lightweight, flexible neural network library written in C++, designed for the specific task of classifying
different types of fruits. This library forms the core part of the larger Fruit Classifier project.

## Features:

- Customizable Neural Network: Create neural networks with varying structures, including different numbers of layers,
  neurons per layer, and activation functions.
- Multiple Activation Functions: Includes several built-in activation functions like step, sign, linear, ReLU, sigmoid,
  and tanh, each with its corresponding derivative for backpropagation.
- Efficient Computation: Optimized for performance, utilizes WebAssembly to allow fast c++ near-native speed.
- Gradient Descent Training: Utilizes gradient descent algorithm for network training.

## Components

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

- **Factory Functions**:
  Located in the ```make``` namespace, these functions allow for the creation of Neurons, Layers, and Networks with
  specific configurations.

## Usage

- To create a neural network:
    - Define Network Structure: Determine the number of layers and neurons in each layer.
    - Select Activation Functions: Choose activation functions for each hidden layer.
    - Create Network: Use the ```make::network``` function to create a network instance.
    - Train the Network: Provide training data and use ```Network::train``` method to train the network.

- Example

```c++
#include "nn/network.h"

int main() {
    // Create a neural network with specific dimensions and activation functions
    // The following network has 3 inputs and 2 hidden layers with [6, 4] neurons respectively
    // and output layer with 2 neurons. And uses sigmoid activation function.
    auto network = nn::make::network({3, 6, 4, 2}, {nn::act::relu, nn::act::sigmoid}, 0.01);

    // Train the network with your data...
    // For input {1, 2, 3} we expect an output {1, 0}
    network.train({1, 2, 3}, {1, 0});
    
    // Now use it to predict outputs...
    auto output = network.predict({2, 3, 1});
    
    return 0;
}
```
