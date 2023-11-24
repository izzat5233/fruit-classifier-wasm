//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_MAKE_H
#define FRUIT_CLASSIFIER_WASM_MAKE_H

#include "nn.h"
#include "act.h"
#include <cstdlib>

/**
 * The Factory SubNamespace
 */
namespace nn::make {

    /**
     * Struct to encapsulate options for creating a Neuron.
     */
    struct NeuronOptions {
        size_t numInputs;
        double lowBound, highBound, bias;

        /**
         * Constructor for NeuronOptions.
         *
         * @param numInputs Number of inputs for the neuron.
         * @param lowBound Lower bound for random weight initialization.
         * @param highBound Upper bound for random weight initialization.
         * @param bias Bias value for the neuron, defaulting to -1.
         */
        NeuronOptions(size_t numInputs, double lowBound, double highBound, double bias = -1)
                : numInputs(numInputs), lowBound(lowBound), highBound(highBound), bias(bias) {}
    };

    /**
     * Struct to encapsulate options for creating a Layer.
     */
    struct LayerOptions {
        size_t numNeurons;
        NeuronOptions neuronOpts;
        act::type activationFunction;

        /**
         * Constructor for LayerOptions.
         *
         * @param numNeurons Number of neurons in the layer.
         * @param neuronOpts Options for creating each neuron in the layer.
         * @param activationFunction Activation function to be used for the layer.
         */
        LayerOptions(size_t numNeurons, NeuronOptions neuronOpts, act::type activationFunction)
                : numNeurons(numNeurons), neuronOpts(neuronOpts), activationFunction(std::move(activationFunction)) {}
    };

    /**
     * Creates a Neuron with the specified options.
     *
     * @param options The NeuronOptions struct containing configuration for the neuron.
     * @return A Neuron object configured as per the provided options.
     */
    Neuron neuron(NeuronOptions options);

    /**
     * Creates a Layer with the specified options.
     *
     * @param options The LayerOptions struct containing configuration for the layer.
     * @return A Layer object configured as per the provided options.
     */
    Layer layer(LayerOptions options);
}

#endif //FRUIT_CLASSIFIER_WASM_MAKE_H
