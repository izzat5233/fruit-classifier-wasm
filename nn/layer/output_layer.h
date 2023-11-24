//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H
#define FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H

#include "layer.h"

#include <utility>

namespace nn::act {
    vd softmax(const vd &x);
}

/**
 * A special layer that uses Softmax activation function.
 */
class nn::OutputLayer : public nn::Layer {
public:
    /**
     * Constructor for the OutputLayer class that initializes the layer with a given set of neurons.
     *
     * @param neurons A vector of Neuron objects.
     */
    explicit OutputLayer(vn neurons);

    /**
     * Processes the inputs through the layer by activating each neuron.
     * Uses Softmax activation function.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] vd process(const vd &inputs) const override;
};

#endif //FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H
