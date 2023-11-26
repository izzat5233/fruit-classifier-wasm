//
// Created by Izzat on 11/25/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NEURON_H
#define FRUIT_CLASSIFIER_WASM_NEURON_H

#include "../nn.h"

class nn::Neuron : public vd_t {
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
};

#endif //FRUIT_CLASSIFIER_WASM_NEURON_H
