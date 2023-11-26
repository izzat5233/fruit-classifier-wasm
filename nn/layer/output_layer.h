//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H
#define FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H

#include "hidden_layer.h"

class nn::OutputLayer : public nn::Layer {
public:
    /**
     * Constructor for the OutputLayer class that initializes the layer with a given core layer.
     *
     * @param layer A core layer which is a collection of neurons.
     */
    explicit OutputLayer(Layer layer);

    /**
     * Processes and cashes the inputs through the layer by activating each neuron.
     * Uses Softmax activation function.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] vd_t activate(const vd_t &inputs) const;
};

#endif //FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H
