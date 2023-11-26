//
// Created by Izzat on 11/26/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_LAYER_H
#define FRUIT_CLASSIFIER_WASM_LAYER_H

#include "../nn.h"
#include "../neuron/neuron.h"

class nn::Layer {
protected:
    vn_t neurons;

    friend Layer make::layer(nn::make::LayerOptions options);

    friend class Network;

public:
    /**
     * Constructor for the Layer class that initializes the layer with a given set of neurons.
     *
     * @param neurons A vector of Neuron objects.
     */
    explicit Layer(vn_t neurons);

    /**
     * @return The number of neurons in the layer
     */
    [[nodiscard]] size_t size() const;

    /**
     * Processes the inputs through the each neuron of the layer.
     * Activation function is not applied yet.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of raw output values from each neuron.
     * Outputs are not activated yet.
     */
    [[nodiscard]] vd_t process(const vd_t &inputs) const;

    /**
     * Calculates the error values for the previous (always hidden) layer in the network.
     * This method is typically used in the backpropagation algorithm to propagate
     * the error backward through the network.
     *
     * The returned error values are not gradient.
     * The derivative activation function must be applied to each value first.
     *
     * @param gradients The vector of error gradients for the current layer.
     * @param previousLayer The previous layer in the neural network.
     * @return A vector of error values for the previous layer.
     */
    [[nodiscard]] vd_t calculateErrors(const vd_t &gradients, const HiddenLayer &previousLayer) const;

    auto begin() noexcept { return neurons.begin(); }

    auto end() noexcept { return neurons.end(); }

    [[nodiscard]] auto begin() const noexcept { return neurons.cbegin(); }

    [[nodiscard]] auto end() const noexcept { return neurons.cend(); }
};

#endif //FRUIT_CLASSIFIER_WASM_LAYER_H
