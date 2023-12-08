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

bool Neuron::operator==(const Neuron &other) const {
    return bias == other.bias &&
           static_cast<const vd_t &>(*this) == static_cast<const vd_t &>(other);
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
    double factor = -1 * alpha * gradient;
    vd_t deltas(inputs.size());
    std::transform(inputs.begin(), inputs.end(), deltas.begin(), [factor](auto y) {
        return y * factor;
    });
    auto biasDelta = factor;
    adjust(deltas, biasDelta);
}

double Neuron::process(const vd_t &inputs) const {
    ASSERT(size() == inputs.size())
    return std::inner_product(begin(), end(), inputs.begin(), 0.0) + bias;
}