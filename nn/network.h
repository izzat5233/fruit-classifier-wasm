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
    double alpha;

public:
    /**
     * Constructs a neural network with a given set of layers and learning rate.
     * The algorithm works well if the last layer is OutputLayer.
     * But no errors occur with another layer.
     *
     * @param hiddenLayers Vector of hidden layers that make up the network.
     * @param outputLayer The OutputLayer of the network.
     * @param alpha Learning rate used in training.
     */
    explicit Network(vl_t hiddenLayers, OutputLayer outputLayer, double alpha);

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
     * Performs full forward & backward propagation given the input-output pair.
     * Cashes all outputs and errors.
     *
     * @param input Vector of input values.
     * @param desired Vector of desired output values.
     */
    void propagate(const vd_t &input, const vd_t &desired);

    /**
     * Trains the neural network on a given input-output pair.
     * Calculates SSE during propagation and returns it.
     *
     * @param input Vector of given input values
     * @param output Vector of expected output values
     * @return The SSE calculated after forward propagation.
     */
    double train(const vd_t &input, const vd_t &output);

    /**
     * Trains the neural network on a given set of input-output pairs.
     * Keeps training until either epochs limit is reached
     * or the error value for all iterations is below the given threshold.
     *
     * @param data Vector of input-output pairs for training.
     * @param epochsLimit A limit to how many epochs the training is allowed to continue.
     * @param errorThreshold A threshold all iteration is below the training stops.
     */
    void train(const vpvd_t &data, std::size_t epochsLimit, double errorThreshold);

    /**
     * Makes predictions based on input data.
     *
     * @param input Vector of input values.
     * @return Predicted output vector.
     */
    [[nodiscard]] vd_t predict(const vd_t &input) const;
};

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_H
