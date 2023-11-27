#include "nn/network.h"

int main() {
    // Create a neural network with specific dimensions and activation functions
    // The following network has 3 inputs and 2 hidden layers with [6, 4] neurons respectively,
    // and output layer with 2 neurons. With initial learning rate 0.01.
    // First hidden layer uses ReLU activation function, second one uses Sigmoid.
    nn::Network network = nn::make::network({3, 6, 4, 2}, {nn::act::relu, nn::act::sigmoid}, 0.01);

    // Train the network with your data...
    // For input {1, 2, 3} we expect an output {1, 0}
    network.train({1, 2, 3}, {1, 0});

    // Now use it to predict outputs...
    // vd_t is a vector of double values.
    nn::vd_t output = network.predict({2, 3, 1});

    return 0;
}