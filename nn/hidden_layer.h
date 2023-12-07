//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H
#define FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H

#include "nn.h"
#include "layer.h"

#include <vector>

class nn::HiddenLayer : public nn::Layer {
private:
    Function function;

    friend class Layer;

public:
    /**
     * Constructor for the HiddenLayer class that initializes the layer with a given core layer
     * and an activation function.
     *
     * @param neurons The neurons of the layer.
     * @param function An activation function to be used for the neurons.
     */
    explicit HiddenLayer(vn_t neurons, Function function);

    [[nodiscard]] vd_t activate(const vd_t &inputs) const override;

    [[nodiscard]] vd_t calculateGradients(const vd_t &intermediateGradients) const override;
};

#endif //FRUIT_CLASSIFIER_WASM_HIDDEN_LAYER_H
