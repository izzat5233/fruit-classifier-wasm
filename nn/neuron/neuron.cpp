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

void Neuron::adjust(const vd_t &weightDeltas, double biasDelta) {
    ASSERT(size() == weightDeltas.size())
    std::transform(begin(), end(), weightDeltas.begin(), begin(), std::plus<>());
    bias += biasDelta;
}

double Neuron::process(const vd_t &inputs) const {
    ASSERT(size() == inputs.size())
    return std::inner_product(begin(), end(), inputs.begin(), 0.0) + bias;
}