#include "nn/nn.h"
#include "nn/make/make.h"
#include "nn/layer/hidden_layer.h"
#include "nn/layer/output_layer.h"
#include "nn/network/network.h"

#include <iostream>

using std::cout, std::cin, std::vector;

int main() {
    vector<double> input = {1, 2};
    vector<double> output = {1, 0};

    auto hiddenLayer = nn::HiddenLayer(nn::make::layer({2, 3}), nn::act::tanh);
    auto outputLayer = nn::OutputLayer(nn::make::layer({3, 2}));
    auto network = nn::Network({hiddenLayer}, outputLayer, 1);

    network.train(input, output);
}
