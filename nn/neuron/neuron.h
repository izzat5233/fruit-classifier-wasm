//
// Created by Izzat on 11/25/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NEURON_H
#define FRUIT_CLASSIFIER_WASM_NEURON_H

#include "../nn.h"

class nn::Neuron {
    vd_t weights;
    double bias;

    friend Neuron make::neuron(make::NeuronOptions options);

    friend class Layer;

public:
    /**
     * Constructor for the Neuron class.
     *
     * @param weights Vector of initial weights.
     * @param threshold Initial bias for the neuron.
     */
    explicit Neuron(vd_t weights, double threshold);

    /**
     * @return The number of weights (or inputs) in the neuron
     */
    [[nodiscard]] size_t size() const;

    /**
     * Adjusts the weights and bias of the neuron.
     *
     * @param weightDeltas Vector of changes to be applied to each weight.
     * @param biasDelta Change applied to the bias.
     */
    void adjust(const vd_t &weightDeltas, double biasDelta);

    /**
     * Calculates the weighted sum of inputs and the bias.
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
