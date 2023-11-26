//
// Created by Izzat on 11/24/2023.
//

#include "neuron.h"
#include "../../util/debug.h"

using namespace nn;

Neuron::Neuron(vd_t weights, double threshold)
        : vd_t(std::move(weights)), bias(threshold) {
    PRINT("Neuron created with " << size() << " weights and bias " << bias)
}

void Neuron::adjust(const vd_t &weightDeltas, double biasDelta) {
    ASSERT(size() == weightDeltas.size())
    *this += weightDeltas;
    bias += biasDelta;
}

double Neuron::process(const vd_t &inputs) const {
    ASSERT(size() == inputs.size())
    auto weightedSum = (*this * inputs).sum();
    return weightedSum + bias;
}