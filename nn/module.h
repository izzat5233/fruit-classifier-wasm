//
// Created by Izzat on 12/17/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_MODULE_H
#define FRUIT_CLASSIFIER_WASM_MODULE_H

#include <optional>
#include "nn.h"
#include "network.h"

class nn::Module {
private:
    /**
     * Manages the normalization and de-normalization of data.
     * It stores the normalized data and the min-max values used for normalization.
     */
    class NormalizedData {
        vvd_t normalized;
        vpd_t minMax;

    public:
        /**
         * Sets and normalizes the data.
         * @param data The data to be normalized and set.
         */
        void set(const vvd_t &data);

        /**
         * Gets the denormalized data.
         * @return The denormalized data.
         */
        [[nodiscard]] vvd_t get() const;

        /**
         * Provides direct access to the normalized data.
         * @return A reference to the normalized data.
         */
        [[nodiscard]] const vvd_t &use() const;

        /**
         * Uses the min-max values stored to normalize the given data.
         * @param original The data to be normalized.
         * @return The normalized data.
         */
        [[nodiscard]] vvd_t normalize(const vvd_t &original) const;

        /**
         * Uses the min-max values stored to de-normalize the given data.
         * @param processed The processed data to be denormalized.
         * @return The denormalized data.
         */
        [[nodiscard]] vvd_t denormalize(const vvd_t &processed) const;
    };

    std::optional<Network> network;
    double alpha = 0.01;

    NormalizedData trainInput;
    NormalizedData trainOutput;

    vvd_t testInput;
    vvd_t testOutput;

public:
    /**
     * Default constructor for Module class.
     */
    explicit Module();

    /**
     * Constructs a Module with an existing network.
     * @param network The neural network to be used in this module.
     */
    explicit Module(Network network);

    /**
     * Sets a new neural network for the module.
     * @param newNetwork The new neural network to be set.
     */
    void setNetwork(Network newNetwork);

    /**
     * Retrieves the weights of the neural network.
     * @return A vector of weight vectors for each layer in the network.
     */
    [[nodiscard]] vvvd_t getWeights() const;

    /**
     * Retrieves the biases of the neural network.
     * @return A vector of biases for each layer in the network.
     */
    [[nodiscard]] vvd_t getBiases() const;

    /**
     * Sets the learning rate for the neural network.
     * @param learningRate The learning rate to be set.
     */
    void setLearningRate(double learningRate);

    /**
     * Gets the current learning rate of the neural network.
     * @return The current learning rate.
     */
    [[nodiscard]] double getLearningRate() const;

    /**
     * Sets the training input data.
     * Normalizes the data and stores it for use in training the network.
     * @param data The training input data to be set.
     */
    void setTrainInput(const vvd_t &data);

    /**
     * Retrieves the original (denormalized) training input data.
     * @return The original training input data.
     */
    [[nodiscard]] vvd_t getTrainInput() const;

    /**
     * Sets the training output data.
     * Normalizes the data and stores it for use in training the network.
     * @param data The training output data to be set.
     */
    void setTrainOutput(const vvd_t &data);

    /**
     * Retrieves the original (denormalized) training output data.
     * @return The original training output data.
     */
    [[nodiscard]] vvd_t getTrainOutput() const;

    /**
     * Sets the testing input data.
     * @param data The testing input data to be set.
     */
    void setTestInput(const vvd_t &data);

    /**
     * Retrieves the testing input data.
     * @return The original testing input data.
     */
    [[nodiscard]] vvd_t getTestInput() const;

    /**
     * Sets the testing output data.
     * @param data The testing output data to be set.
     */
    void setTestOutput(const vvd_t &data);

    /**
     * Retrieves the testing output data.
     * @return The original testing output data.
     */
    [[nodiscard]] vvd_t getTestOutput() const;

    /**
     * Trains the neural network for one epoch using the provided training data.
     * The function calculates and returns the average error over all training instances.
     *
     * @return The average training error for the epoch.
     */
    double train();

    /**
      * Evaluates the neural network's performance on the testing dataset for one epoch.
      * This function iterates through all testing data without modifying network weights.
      * Calculates and returns the average error over all testing instances.
      *
      * @return The average testing error for the epoch.
      */
    [[nodiscard]] double test() const;

    /**
     * Repeatedly trains the neural network for a specified number of epochs.
     * Each epoch involves training the network on the entire training dataset.
     * Returns a vector of average training errors for each epoch.
     *
     * @param epochs The number of epochs to train the network.
     * @return A vector of average training errors, one for each epoch.
     */
    vd_t train(std::size_t epochs);

    /**
     * Repeatedly tests the neural network for a specified number of epochs.
     * Each epoch involves evaluating the network on the entire testing dataset.
     * Returns a vector of average testing errors for each epoch.
     *
     * @param epochs The number of epochs to test the network.
     * @return A vector of average testing errors, one for each epoch.
     */
    [[nodiscard]] vd_t test(std::size_t epochs) const;

    /**
     * Conducts a combined training and testing process for a specified number of epochs.
     * In each epoch, the network is first trained and then tested.
     * The function returns a vector of pairs, each containing the average training and testing errors for an epoch.
     *
     * @param epochs The number of epochs to train and test the network.
     * @return A vector of pairs of average training and testing errors for each epoch.
     */
    [[nodiscard]] vpd_t trainAndTest(std::size_t epochs);

    /**
     * Predicts the outputs for the testing dataset.
     * Utilizes the trained network to make predictions on the test inputs.
     * The function returns the predicted outputs, which are denormalized for interpretation.
     *
     * @return A vector of vectors containing the predicted outputs for the test data.
     */
    [[nodiscard]] vvd_t predict() const;

    /**
     * Predicts the outputs for externally provided input data.
     * This function uses the trained network to make predictions on the given input data.
     * Normalization process happens internally. You should provide denormalized data.
     *
     * @param inputData The external input data for which predictions are to be made.
     * @return A vector of vectors containing the predicted outputs for the input data.
     */
    [[nodiscard]] vvd_t predict(const vvd_t &inputData) const;
};

#endif //FRUIT_CLASSIFIER_WASM_MODULE_H
