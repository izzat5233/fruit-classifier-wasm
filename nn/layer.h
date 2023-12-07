//
// Created by Izzat on 11/26/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_LAYER_H
#define FRUIT_CLASSIFIER_WASM_LAYER_H

#include "nn.h"
#include "neuron.h"

class nn::Layer : public vn_t {
protected:
    vd_t cash;

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
     * Processes the inputs through the layer by activating each neuron.
     * Activation function is applied to every output.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    [[nodiscard]] virtual vd_t activate(const vd_t &inputs) const = 0;

    /**
     * Processes and caches the inputs through the layer by activating each neuron.
     * This method caches the output, which is useful for subsequent backpropagation steps
     * during training.
     *
     * @param inputs A vector of input values to the layer.
     * @return A vector of output values from each neuron.
     */
    vd_t activateAndCache(const vd_t &inputs);

    /**
     * Propagates the error backward from the current layer to the previous layer in the network.
     * This method computes the preliminary component of the gradients for the previous layer
     * by calculating the weighted sum of the current layer's gradients and each neuron's weights.
     *
     * This computation is a fundamental part of the backpropagation algorithm and is used for
     * adjusting the weights in the network. It effectively distributes the error from the current
     * layer back to the previous layer, taking into account the contribution of each neuron's weights
     * to the overall error.
     *
     * Note: This method only computes the weighted sum of gradients and does not apply the derivative
     * of the activation function. The computed values are intermediate gradients, and the derivative
     * of the previous layer's activation function should be applied externally to obtain the final gradients.
     *
     * @param gradients A vector of error gradients from the current layer. These gradients are typically
     * the differences between the predicted output and the actual output of the network.
     * @return A vector representing the preliminary gradients for the previous layer, calculated as a
     * weighted sum of the current layer's gradients and neuron weights.
     */
    [[nodiscard]] vd_t propagateErrorBackward(const vd_t &gradients) const;

    /**
     * Abstract method for calculating the gradients for the layer. This method is designed to be
     * overridden in derived classes. It takes the intermediate gradients, which are the outputs of
     * the `propagateErrorBackward` method from the next layer in the network, and applies the derivative of the
     * activation function to these intermediate gradients to produce the final gradients for the current layer.
     *
     * This function is a crucial part of the backpropagation algorithm, enabling the calculation of
     * gradients that are necessary for updating the weights of the neurons in the network. The derivative
     * applied to the intermediate gradients allows the error signal to be modulated according to the
     * non-linearities introduced by the activation function of the layer.
     *
     * Note: this method uses the outputs cashed by the latest `activate` method call.
     *
     * @param intermediateGradients A vector of gradients obtained from the `propagateErrorBackward` method of the
     * next layer. These gradients are pre-derivative and need to be processed further.
     * @return A vector of final gradients for the current layer after applying the derivative of
     * the activation function.
     */
    [[nodiscard]] virtual vd_t calculateGradients(const vd_t &intermediateGradients) const = 0;

    /**
     * Static utility method for calculating gradients based on two consecutive layers and
     * the gradients of the latter layer. It uses the `propagateErrorBackward` method of the first layer (`l1`)
     * and the `calculateGradients` method of the second layer (`l2`) to compute the gradients.
     *
     * This method is particularly useful in chaining gradient computations across layers during
     * the backpropagation phase of a neural network. By utilizing the weighted sum from `propagateErrorBackward`
     * and the gradient calculation logic in `calculateGradients`, it effectively bridges the gradient
     * computation between two layers.
     *
     *
     * @param l1 The first layer (usually closer to the input).
     * @param l2 The second layer (usually closer to the output).
     * @param l2Gradients Gradients calculated for the second layer (`l2`).
     * @return A vector of gradients for the first layer (`l1`), ready for backpropagation.
     */
    [[nodiscard]] static vd_t calculateGradients(const Layer &l1, const Layer &l2, const vd_t &l2Gradients);
};

#endif //FRUIT_CLASSIFIER_WASM_LAYER_H
