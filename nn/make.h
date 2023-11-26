//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_MAKE_H
#define FRUIT_CLASSIFIER_WASM_MAKE_H

#include "nn.h"
#include <cstdlib>
#include <utility>

namespace nn::make {
    /**
     * Struct to encapsulate options for creating a Neuron.
     */
    struct NeuronOptions {
        size_t numInputs;
        double lowBound, highBound;

        /**
         * Constructor for NeuronOptions.
         * Weights and bias are given random values between the given boundaries.
         *
         * @param numInputs Number of inputs for the neuron.
         * @param lowBound Lower bound for random weight initialization.
         * @param highBound Upper bound for random weight initialization.
         */
        NeuronOptions(size_t numInputs, double lowBound, double highBound)
                : numInputs(numInputs), lowBound(lowBound), highBound(highBound) {}
    };

    /**
     * Struct to encapsulate options for creating a Layer.
     */
    struct LayerOptions {
        size_t numInputs, numNeurons;

        /**
         * Constructor for LayerOptions.
         * Neurons weights and bias are given random values in the range:
         * [-numNeurons / 2.4, +numNeurons / 2.4].
         *
         * @param numInputs Number of inputs for the neurons.
         * @param numNeurons Number of neurons for the layer.
         */
        LayerOptions(size_t numInputs, size_t numNeurons)
                : numInputs(numInputs), numNeurons(numNeurons) {}
    };
}

#endif //FRUIT_CLASSIFIER_WASM_MAKE_H
