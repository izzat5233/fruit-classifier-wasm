//
// Created by Izzat on 11/24/2023.
//

#include "neuron.h"
#include "../../util/debug.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace nn;

Neuron::Neuron(vd initialWeights, double threshold)
        : weights(std::move(initialWeights)), bias(threshold) {
    PRINT("Neuron created with " << weights.size() << " weights and bias " << bias)
}

void Neuron::adjustWeights(const vd &deltas) {
    ASSERT(weights.size() == deltas.size())
    std::transform(weights.begin(), weights.end(), deltas.begin(), weights.begin(), std::plus<>());
}

double Neuron::process(const vd &inputs) const {
    ASSERT(inputs.size() == weights.size())
    return std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0) + bias;
}

Neuron make::neuron(make::NeuronOptions options) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    auto [n, l, h, b] = options;
    std::uniform_real_distribution<> dist(l, h);

    vd weights;
    for (size_t i = 0; i < n; ++i) { weights.push_back(dist(gen)); }

    PRINT_ITER("Created neuron with weights:", weights)
    return Neuron(weights, b);
}
