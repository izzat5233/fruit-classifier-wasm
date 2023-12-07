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
     * @param neurons The neurons of the layer.
     */
    explicit OutputLayer(vn_t neurons);

    [[nodiscard]] vd_t activate(const vd_t &inputs) const override;

    [[nodiscard]] vd_t calculateGradients(const vd_t &intermediateGradients) const override;
};

#endif //FRUIT_CLASSIFIER_WASM_OUTPUT_LAYER_H
