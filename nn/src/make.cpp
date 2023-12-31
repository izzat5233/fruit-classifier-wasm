//
// Created by Izzat on 11/26/2023.
//

#include "nn.h"
#include "neuron.h"
#include "network.h"

#include <random>
#include <cassert>

using namespace nn;

Neuron make::neuron(const ui_t &numInputs, double lowBound, double highBound) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(lowBound, highBound);

    vd_t weights(numInputs);
    for (auto &i: weights) { i = dist(gen); }

    return Neuron(weights, dist(gen));
}

vn_t make::layer(const ui_t &numInputs, const ui_t &numNeurons, double rangeFactor) {
    assert(rangeFactor > 0);
    auto lowBound = -numNeurons / rangeFactor;
    auto highBound = numNeurons / rangeFactor;

    vn_t neurons;
    neurons.reserve(numNeurons);
    for (ui_t i = 0; i < numNeurons; ++i) {
        neurons.push_back(make::neuron(numInputs, lowBound, highBound));
    }

    return neurons;
}

Network make::network(const vi_t &dimensions, const vf_t &functions, loss::function_t lossFunction) {
    auto n = static_cast<ui_t>(dimensions.size());
    assert(n == dimensions.size());

    vl_t layers;
    for (ui_t i = 1; i < n - 1; ++i) {
        layers.emplace_back(make::layer(dimensions[i - 1], dimensions[i]), functions[i - 1]);
    }

    OutputLayer outputLayer(make::layer(dimensions[n - 2], dimensions[n - 1]));
    return Network(layers, outputLayer, lossFunction);
}

Network make::network(const vi_t &dimensions, const act::Function &function, loss::function_t lossFunction) {
    return make::network(dimensions, vf_t(dimensions.size() - 2, function), lossFunction);
}