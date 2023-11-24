//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_LAYER_H
#define FRUIT_CLASSIFIER_WASM_LAYER_H

#include "../nn.h"
#include "../make.h"
#include "../neuron/neuron.h"
#include "../act.h"
#include <vector>

/**
 * Represents a layer in a neural network. A layer is a collection of neurons that
 * processes inputs and produces outputs based on the defined activation function.
 * All neurons in the layer are processing using the same activation function
 */
class nn::Layer {
private:
    vn neurons;
    act::type function;

    friend Layer make::layer(make::LayerOptions options);

protected:

    /**
     * Processes the inputs through the each neuron.
     * Doesn't apply the activation function yet.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] vd calculateOutputs(const vd &inputs) const;

public:
    /**
     * Constructor for the Layer class that initializes the layer with a given set of neurons
     * and an activation function.
     *
     * @param neurons A vector of Neuron objects.
     * @param function An activation function to be used for the neurons.
     */
    explicit Layer(vn neurons, act::type function);

    /**
     * Processes the inputs through the layer by activating each neuron.
     * Activation function is applied to every neuron.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] virtual vd process(const vd &inputs) const;

    auto begin() noexcept { return neurons.begin(); }

    auto end() noexcept { return neurons.end(); }

    [[nodiscard]] auto begin() const noexcept { return neurons.cbegin(); }

    [[nodiscard]] auto end() const noexcept { return neurons.cend(); }
};

#endif //FRUIT_CLASSIFIER_WASM_LAYER_H
