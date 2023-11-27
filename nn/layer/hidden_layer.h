//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H
#define FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H

#include "../nn.h"
#include "layer.h"

#include <vector>

class nn::HiddenLayer : public nn::Layer {
private:
    act::Function function;

    friend class Layer;

public:
    /**
     * Constructor for the HiddenLayer class that initializes the layer with a given core layer
     * and an activation function.
     *
     * @param layer A core layer which is a collection of neurons.
     * @param function An activation function to be used for the neurons.
     */
    explicit HiddenLayer(Layer layer, act::Function function);

    /**
     * Processes the inputs through the layer by activating each neuron.
     * Activation function is applied to every output.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] vd_t activate(const vd_t &inputs) const;
};

#endif //FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H
