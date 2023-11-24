#include <iostream>
#include "lib/nn/layer.h"

using std::cout, std::cin, std::vector;

int main() {
    vector<double> inputs = {1, 2};
    auto neuronOptions = nn::make::NeuronOptions{inputs.size(), -1, 1};
    auto layerOptions = nn::make::LayerOptions{4, neuronOptions, nn::act::linear};
    auto layer = nn::make::layer(layerOptions);
    auto res = layer.process(inputs);
}
