//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NEURON_H
#define FRUIT_CLASSIFIER_WASM_NEURON_H

#include "../nn.h"
#include "../make.h"
#include <vector>

/**
 * Class representing a single neuron in a neural network.
 * This class encapsulates the neuron's weights and bias,
 * and provides functionality for computing the neuron's output.
 */
class nn::Neuron {
    vd_t weights;
    double bias;

    friend Neuron make::neuron(make::NeuronOptions options);

public:
    /**
     * Constructor for the Neuron class.
     *
     * @param initialWeights Vector of initial weights.
     * @param threshold Initial bias for the neuron, defaulting to -1.
     */
    explicit Neuron(vd_t initialWeights, double threshold);

    /**
     * Adjusts the weights of the neuron.
     *
     * @param deltas Vector of changes to be applied to each weight.
     */
    void adjust(const vd_t &deltas);

    /**
     * @return The number of weights (or inputs) in the neuron
     */
    [[nodiscard]] size_t size() const;

    /**
     * Calculates the weighted sum of inputs and the bias.
     * It does not apply any activation function yet.
     *
     * @param inputs Vector of input values.
     * @return The weighted sum.
     */
    [[nodiscard]] double process(const vd_t &inputs) const;

    auto begin() noexcept { return weights.begin(); }

    auto end() noexcept { return weights.end(); }

    [[nodiscard]] auto begin() const noexcept { return weights.cbegin(); }

    [[nodiscard]] auto end() const noexcept { return weights.cend(); }
};

#endif //FRUIT_CLASSIFIER_WASM_NEURON_H
