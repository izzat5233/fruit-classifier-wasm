//
// Created by Izzat on 11/26/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_LAYER_H
#define FRUIT_CLASSIFIER_WASM_LAYER_H

#include "nn.h"
#include "neuron.h"

class nn::Layer : public vn_t {
private:
    friend class Network;

public:
    /**
     * Constructor for the Layer class that initializes the layer with a given set of neurons.
     *
     * @param n A vector of Neuron objects.
     */
    explicit Layer(vn_t n);

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
     * Calculates a component of the error gradient values for the previous (always hidden) layer in the network.
     * This method is typically used in the backpropagation algorithm to propagate
     * the gradient errors backward through the network. It computes a weighted sum
     * of the current layer's gradients and each neuron's weights.
     *
     * Note: This method does not apply the derivative of the previous layer's activation
     * function to the calculated sum. The caller is responsible for applying this derivative
     * to obtain the final gradient errors for the previous layer.
     *
     * The gradient values can be obtained by applying the derivative on the outputs of the layer,
     * then multiplying these values with the weighted sum values obtained from this method.
     *
     * @param w A vector of error gradients for the current layer.
     * @param acc A previous layer in the neural network.
     * @return A vector representing a weighted sum of the current layer's gradients and neuron weights.
     */
    [[nodiscard]] vd_t backPropagate(const vd_t &w, const HiddenLayer &acc) const;
};

#endif //FRUIT_CLASSIFIER_WASM_LAYER_H
