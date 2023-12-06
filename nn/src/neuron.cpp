//
// Created by Izzat on 11/24/2023.
//

#include "neuron.h"
#include "../../util/debug.h"

#include <algorithm>
#include <numeric>

using namespace nn;

Neuron::Neuron(vd_t weights, double threshold)
        : vd_t(std::move(weights)), bias(threshold) {
    PRINT("Neuron created with " << size() << " weights and bias " << bias)
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::adjust(const vd_t &weightDeltas, double biasDelta) {
    ASSERT(size() == weightDeltas.size())
    std::transform(begin(), end(), weightDeltas.begin(), begin(), std::plus<>());
    bias += biasDelta;
    PRINT_ITER("Neuron adjusted weights to: ", *this)
    PRINT("Neuron adjusted bias to: " << bias)
}

void Neuron::adjust(const vd_t &inputs, double gradient, double alpha) {
    vd_t deltas(inputs.size());
    std::transform(inputs.begin(), inputs.end(), deltas.begin(), [alpha, gradient](auto y) {
        return alpha * y * gradient;
    });
    auto biasDelta = alpha * -1 * gradient;
    adjust(deltas, biasDelta);
}

double Neuron::process(const vd_t &inputs) const {
    ASSERT(size() == inputs.size())
    return std::inner_product(begin(), end(), inputs.begin(), 0.0) + bias;
}