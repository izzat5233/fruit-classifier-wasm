#include <iostream>
#include "nn/nn.h"
#include "nn/make.h"
#include "nn/neuron/neuron.h"
#include "nn/layer/hidden_layer.h"
#include "nn/layer/output_layer.h"

using std::cout, std::cin, std::vector;

int main() {
    vector<double> inputs = {1, 2};

    auto hiddenLayer = nn::HiddenLayer(nn::make::layer({2, 3}), nn::act::tanh.fun);
    auto outputLayer = nn::OutputLayer(nn::make::layer({3, 3}));

    auto output = outputLayer.activate(hiddenLayer.activate(inputs));
    cout << "Result: ";
    for (auto i: output) { cout << i << ' '; }
}
