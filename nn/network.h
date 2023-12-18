//
// Created by Izzat on 11/27/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NETWORK_H
#define FRUIT_CLASSIFIER_WASM_NETWORK_H

#include "nn.h"
#include "hidden_layer.h"
#include "output_layer.h"

class nn::Network {
private:
    const std::size_t size;
    vl_t layers;
    OutputLayer outputLayer;
    loss::function_t lossFunction;

public:
    /**
     * Constructs a neural network with a given set of layers and loss function,.
     * The algorithm works well if the last layer is OutputLayer.
     * But no errors occur with another layer.
     *
     * @param hiddenLayers Vector of hidden layers that make up the network.
     * @param outputLayer The OutputLayer of the network.
     * @param lossFunction The loss function used in backpropagation
     */
    explicit Network(vl_t hiddenLayers, OutputLayer outputLayer, loss::function_t lossFunction);

    /**
     * @return The number of layers in the network, including the output layer.
     */
    [[nodiscard]] std::size_t getSize() const;

    /**
     * Provides access to a specific layer in the network.
     *
     * @param index The index of the layer to access.
     * @return A reference to the requested layer.
     */
    Layer &get(std::size_t index);

    /**
     * Provides access to a specific layer in the network.
     *
     * @param index The index of the layer to access.
     * @return A const reference to the requested layer.
     */
    [[nodiscard]] const Layer &get(std::size_t index) const;

    /**
     * Provides access to a specific layer in the network in reversed order.
     *
     * @param index The reverse index of the layer to access.
     * @return A reference to the requested layer.
     */
    Layer &rget(std::size_t index);

    /**
     * Provides access to a specific layer in the network in reversed order.
     *
     * @param index The reverse index of the layer to access.
     * @return A const reference to the requested layer.
     */
    [[nodiscard]] const Layer &rget(std::size_t index) const;

    /**
     * Forward-propagates the inputs vector through the network.
     * All layers are activated and all their outputs are cashed.
     *
     * @param input Vector of input values.
     * @return The calculated output values.
     */
    vd_t forwardPropagate(const vd_t &input);

    /**
     * Backward-propagates the inputs vector through the network.
     * The given desired outputs must be one-hot coded for the algorithm to work well.
     *
     * @param desired Vector of desired output values.
     */
    void backwardPropagate(const vd_t &desired);

    /**
     * Trains the neural network on a given input-output pair.
     * A call to this method represents a single iteration on the data.
     *
     * @param input Vector of given input values
     * @param output Vector of expected output values
     * @param alpha Learning rate
     * @return The outputs error calculated by the lossFunction.
     */
    double train(const vd_t &input, const vd_t &output, double alpha);

    /**
     * Tests the neural network on a given input-output pair.
     * A call to this method represents a single iteration on the data.
     *
     * @param input Vector of given input values
     * @param output Vector of expected output values
     * @return The outputs error calculated by the lossFunction.
     */
    [[nodiscard]] double test(const vd_t &input, const vd_t &output) const;

    /**
     * Makes predictions based on input data.
     *
     * @param input Vector of input values.
     * @return Predicted output vector.
     */
    [[nodiscard]] vd_t predict(const vd_t &input) const;
};

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_H
