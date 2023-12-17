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
     * Normalization Manager.
     */
    class NormalizedData {
        vvd_t normalized;
        vpd_t minMax;

    public:
        void set(const vvd_t &data);

        [[nodiscard]] vvd_t get() const;

        [[nodiscard]] vvd_t get(const vvd_t &processed) const;

        [[nodiscard]] const vvd_t &use() const;
    };

    std::optional<Network> network;
    NormalizedData trainInput;
    NormalizedData trainOutput;
    NormalizedData testInput;
    NormalizedData testOutput;

public:

    explicit Module();

    explicit Module(Network network);

    void setNetwork(Network newNetwork);

    [[nodiscard]] const std::optional<Network> &getNetwork() const;

    void setTrainInput(const vvd_t &data);

    [[nodiscard]] vvd_t getTrainInput() const;

    void setTrainOutput(const vvd_t &data);

    [[nodiscard]] vvd_t getTrainOutput() const;

    void setTestInput(const vvd_t &data);

    [[nodiscard]] vvd_t getTestInput() const;

    void setTestOutput(const vvd_t &data);

    [[nodiscard]] vvd_t getTestOutput() const;

    /**
     * Trains the neural network for one epoch.
     * Calculates the average error of all iterations and returns it.
     *
     * @return The average error result of all iterations.
     */
    double train();

    /**
     * Tests the neural network for one epoch.
     * Calculates the average error of all iterations and returns it.
     *
     * @return The average error result of all iterations.
     */
    [[nodiscard]] double test() const;

    /**
     * Trains the neural network for a number of epoch.
     * Calculates the average error for each epoch and returns them all.
     *
     * @param epochs Number of training epochs.
     * @return The average error results of each epoch.
     */
    vd_t train(std::size_t epochs);

    /**
     * Tests the neural network for a number of epoch.
     * Calculates the average error for each epoch and returns them all.
     *
     * @param epochs Number of training epochs.
     * @return The average error results of each epoch.
     */
    [[nodiscard]] vd_t test(std::size_t epochs) const;

    /**
     * Predicts test data outputs.
     *
     * @param inputs input data vectors.
     * @return predicted output vectors.
     */
    [[nodiscard]] vvd_t predict() const;

    /**
     * Trains & Tests simultaneously for a number of epochs.
     * It trains for one epoch then tests for one epoch and repeats.
     *
     * @param epochs Number of training and testing epochs.
     * @return The average train error and test error pairs for all epochs.
     */
    [[nodiscard]] vpd_t trainAndTest(std::size_t epochs);
};

#endif //FRUIT_CLASSIFIER_WASM_MODULE_H
