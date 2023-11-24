//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NEURON_H
#define FRUIT_CLASSIFIER_WASM_NEURON_H

#include <utility>
#include <vector>
#include <numeric>
#include <functional>

#include "debug.h"

namespace nn {
    class Neuron;
}

/**
 * Class representing a single neuron in a neural network.
 * This class encapsulates the neuron's weights and bias,
 * and provides functionality for computing the neuron's output.
 */
class nn::Neuron {
private:
    using vdl = std::vector<double>;
    vdl weights;
    double bias;

public:
    /**
     * Constructor for the Neuron class.
     *
     * @param initialWeights Vector of initial weights.
     * @param threshold Initial bias for the neuron, defaulting to -1.
     */
    explicit Neuron(vdl initialWeights, double threshold = -1)
            : weights(std::move(initialWeights)), bias(threshold) {
        PRINT("Neuron created with " << weights.size() << " weights and bias " << bias)
    }

    /**
     * Adjusts the weights of the neuron.
     *
     * @param deltas Vector of changes to be applied to each weight.
     */
    void adjustWeights(const vdl &deltas) {
        ASSERT(weights.size() == deltas.size())
        std::transform(weights.begin(), weights.end(), deltas.begin(), weights.begin(), std::plus<>());
    }

    /**
     * Calculates the weighted sum of inputs and the bias.
     *
     * @param inputs Vector of input values.
     * @return The weighted sum.
     */
    [[nodiscard]] double sumInputs(const vdl &inputs) const {
        ASSERT(inputs.size() == weights.size())
        return std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0) + bias;
    }

    auto begin() noexcept { return weights.begin(); }

    auto end() noexcept { return weights.end(); }

    [[nodiscard]] auto begin() const noexcept { return weights.cbegin(); }

    [[nodiscard]] auto end() const noexcept { return weights.cend(); }
};

#endif //FRUIT_CLASSIFIER_WASM_NEURON_H
