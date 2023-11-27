//
// Created by Izzat on 11/26/2023.
//

#include "../nn/nn.h"
#include "../nn/neuron.h"
#include "../nn/layer.h"
#include "../nn/network.h"
#include "../util/debug.h"

#include <random>

using namespace nn;

Neuron make::neuron(const ui_t &numInputs, double lowBound, double highBound) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(lowBound, highBound);

    vd_t weights(numInputs);
    for (auto &i: weights) { i = dist(gen); }

    PRINT_ITER("Random weights:", weights)
    return Neuron(weights, dist(gen));
}

Layer make::layer(const ui_t &numInputs, const ui_t &numNeurons, double rangeFactor) {
    ASSERT(rangeFactor > 0)
    auto lowBound = -numNeurons / rangeFactor;
    auto highBound = numNeurons / rangeFactor;

    vn_t neurons;
    neurons.reserve(numNeurons);
    for (ui_t i = 0; i < numNeurons; ++i) {
        neurons.push_back(make::neuron(numInputs, lowBound, highBound));
    }

    return Layer(neurons);
}

Network make::network(const vi_t &dimensions, const vf_t &functions, double alpha) {
    auto n = static_cast<ui_t>(dimensions.size());
    ASSERT(n == dimensions.size())

    vl_t layers;
    for (ui_t i = 1; i < n - 1; ++i) {
        layers.emplace_back(make::layer(dimensions[i - 1], dimensions[i]), functions[i - 1]);
    }

    OutputLayer outputLayer(make::layer(dimensions[n - 2], dimensions[n - 1]));
    return Network(layers, outputLayer, alpha);
}

Network make::network(const vi_t &dimensions, const Function &function, double alpha) {
    return make::network(dimensions, vf_t(dimensions.size() - 2, function), alpha);
}