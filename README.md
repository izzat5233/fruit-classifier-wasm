# Fruism | Fruit Classifier Neural Network in WASM

The Fruism is a lightweight, flexible neural network library written in C++, designed for the specific task of
classifying different types of fruits. This library forms the core part of the larger Fruit Classifier project.

## Features:

- *Customizable Neural Network*: Create neural networks with varying structures, including different numbers of layers,
  neurons per layer, and activation functions.
- *Multiple Activation Functions*: Includes several built-in activation functions like step, sign, linear, ReLU,
  sigmoid,
  and tanh, each with its corresponding derivative for backpropagation.
- *Efficient Computation*: Optimized for performance, utilizes WebAssembly to allow fast c++ near-native speed.
- *Gradient Descent Training*: Utilizes gradient descent algorithm for network training.

## Components

- **[```nn```](nn/nn.h)**: Main namespace containing all components.
  Check [this](nn/nn.h) header for documentation.

- **[```Neuron```](nn/neuron.h)**: Represents a single neuron, encapsulating weights and bias.
  Provides functionality for computing neuron's output.

- **Layer Types**:
    - **[```Layer```](nn/layer.h)**: Base class for layers in the neural network.
    - **[```HiddenLayer```](nn/hidden_layer.h)**: Represents a hidden layer in the network.
    - **[```OutputLayer```](nn/output_layer.h)**: Special layer type using Softmax activation.

- **[```Network```](nn/network.h)**:Represents the entire neural network, a collection of layers.
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

## Development

To get started with development:

1. **Emscripten Installation**: Make sure Emscripten is installed and properly configured on your system. Follow the
   installation guide on the official [Emscripten website](https://emscripten.org/docs/getting_started/downloads.html).

2. **Configure Build Profiles**:
    - **Release Profile**: For building the final WebAssembly version of the project, use the Release profile. This
      includes Emscripten's toolchain for WebAssembly compilation. Compiled wasm files will be located
      at `/web/static/assets/wasm`
      afterward to be used by hugo server.
        ```sh
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=path/to/emscripten/cmake/Modules/Platform/Emscripten.cmake -B build/release -S .
        ```
    - **Debug Profile**: For development and testing, use the Debug profile. This profile is configured to run all the
      tests in the `test` directory.
        ```sh
        cmake -DCMAKE_BUILD_TYPE=Debug -B build/debug -S .
        ```

3. **Build the Project**:
    - Navigate to the appropriate build directory (`build/debug` or `build/release`) and build the project:
        ```sh
        cmake --build .
        ```

4. **Run Tests** (Debug profile only):
    - In the Debug profile, execute the tests to ensure everything is functioning correctly:
        ```sh
        ctest
        ```

5. **Web Integration** (Release profile only):
    - For the Release profile, include the generated JavaScript and WebAssembly files in your web project. Use the
      Emscripten Module API for interaction with the compiled code.
    - All wasm files are generated at `/web/static/wasm`. `web/static` directory is served as-is by hugo server.