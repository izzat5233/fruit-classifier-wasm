#include <iostream>
#include "nn/layer/layer.h"
#include "nn/layer/output_layer.h"

using std::cout, std::cin, std::vector;

int main() {
    vector<double> inputs = {1, 2};
    auto neuronOptions = nn::make::NeuronOptions{inputs.size(), -1, 1};

    vector<nn::Neuron> neurons;
    neurons.reserve(4);
    for (int i = 0; i < 4; ++i) {
        neurons.push_back(nn::make::neuron(neuronOptions));
    }

    auto layer = nn::OutputLayer(neurons);
    auto res = layer.process(inputs);
}
