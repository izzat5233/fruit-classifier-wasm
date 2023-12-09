//
// Created by Izzat on 11/24/2023.
//

#include "neuron.h"

#include <algorithm>
#include <numeric>
#include <cassert>

using namespace nn;

Neuron::Neuron(vd_t weights, double threshold) : vd_t(std::move(weights)), bias(threshold) {}

double Neuron::getBias() const {
    return bias;
}

void Neuron::adjust(const vd_t &weightDeltas, double biasDelta) {
    assert(size() == weightDeltas.size());
    std::transform(begin(), end(), weightDeltas.begin(), begin(), std::plus<>());
    bias += biasDelta;
}

void Neuron::adjust(const vd_t &inputs, double gradient, double alpha) {
    double factor = -1 * alpha * gradient;
    vd_t deltas(inputs.size());
    std::transform(inputs.begin(), inputs.end(), deltas.begin(), [factor](auto y) {
        return y * factor;
    });
    auto biasDelta = factor;
    adjust(deltas, biasDelta);
}

double Neuron::process(const vd_t &inputs) const {
    assert(size() == inputs.size());
    return std::inner_product(begin(), end(), inputs.begin(), 0.0) + bias;
}