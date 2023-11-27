//
// Created by Izzat on 11/25/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NEURON_H
#define FRUIT_CLASSIFIER_WASM_NEURON_H

#include "../nn.h"

class nn::Neuron : public vd_t {
    double bias;

    friend Neuron make::neuron(const ui_t &numInputs, const double &lowBound, const double &highBound);

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
     * Uses the gradient and learning rate to calculate all the deltas.
     * Then adjust the weights and bias of the neuron.
     *
     * Gradient value is one that resulted from backpropagation
     * in which the given inputs were passed to this neuron.
     *
     * @param inputs Vector of input values passed to the neuron
     * @param gradient Gradient error value
     * @param alpha Learning rate
     */
    void adjust(const vd_t &inputs, double gradient, double alpha);

    /**
     * Calculates the weighted sum of inputs and the bias.
     *
     * @param inputs Vector of input values.
     * @return The weighted sum.
     */
    [[nodiscard]] double process(const vd_t &inputs) const;
};

#endif //FRUIT_CLASSIFIER_WASM_NEURON_H
