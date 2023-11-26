//
// Created by Izzat on 11/26/2023.
//

#include "make.h"
#include "../neuron/neuron.h"
#include "../layer/layer.h"
#include "../../util/debug.h"

#include <random>

using namespace nn;

Neuron make::neuron(make::NeuronOptions options) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    auto [n, l, h] = options;
    std::uniform_real_distribution<> dist(l, h);

    vd_t weights(n);
    for (auto &i: weights) { i = dist(gen); }

    PRINT_ITER("Random weights:", weights)
    return Neuron(weights, dist(gen));
}

Layer make::layer(nn::make::LayerOptions options) {
    auto [numInputs, numNeurons] = options;
    make::NeuronOptions neuronOptions(numInputs, -numNeurons / 2.4, numNeurons / 2.4);

    vn_t neurons;
    neurons.reserve(numNeurons);
    for (size_t i = 0; i < numNeurons; ++i) { neurons.push_back(make::neuron(neuronOptions)); }

    return Layer(neurons);
}