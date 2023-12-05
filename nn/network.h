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
    vl_t layers;
    OutputLayer outputLayer;
    double alpha;
    std::size_t size;
    vvd_t y_cash;
    vvd_t e_cash;

    /**
     * Forward-propagates the inputs vector through the network.
     * Uses the activation function for each layer to activate the outputs.
     * Cashes all outputs on the way.
     *
     * @param input Vector of input values.
     */
    void forwardPropagate(const vd_t &input);

    /**
     * Backward-propagates the inputs vector through the network.
     * Uses the derivative function for each layer to perform gradient descent.
     * The given desired outputs must be one-hot coded for the algorithm to work.
     * Cashes all errors on the way back.
     *
     * @param output Vector of desired output values.
     */
    void backwardPropagate(const vd_t &output);

    /**
     * Performs full forward & backward propagation given the input-output pair.
     * Cashes all outputs and errors.
     *
     * @param input Vector of input values.
     * @param output Vector of desired output values.
     */
    void propagate(const vd_t &input, const vd_t &output);

public:
    /**
     * Constructs a neural network with a given set of layers and a learning rate.
     *
     * @param layers Vector of hidden layers that make up the network.
     * @param outputLayer The OutputLayer of the network.
     * @param alpha Learning rate used in training.
     */
    explicit Network(vl_t layers, OutputLayer outputLayer, double alpha);

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
