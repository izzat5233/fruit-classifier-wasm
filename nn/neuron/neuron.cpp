//
// Created by Izzat on 11/24/2023.
//

#include "neuron.h"
#include "../make.h"
#include "../../util/debug.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace nn;

Neuron::Neuron(vd_t weights, double threshold)
        : weights(std::move(weights)), bias(threshold) {
    PRINT("Neuron created with " << this->weights.size() << " weights and bias " << bias)
}

nn::size_t Neuron::size() const {
    return static_cast<size_t>(weights.size());
}

void Neuron::adjust(const vd_t &weightDeltas, double biasDelta) {
    ASSERT(weights.size() == weightDeltas.size())
    std::transform(weights.begin(), weights.end(), weightDeltas.begin(), weights.begin(), std::plus<>());
    bias += biasDelta;
}

double Neuron::process(const vd_t &inputs) const {
    ASSERT(inputs.size() == weights.size())
    return std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0) + bias;
}

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