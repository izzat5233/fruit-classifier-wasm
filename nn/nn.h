//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NN_H
#define FRUIT_CLASSIFIER_WASM_NN_H

#include <vector>

/**
 * Neural Network Namespace
 */
namespace nn {
    class Neuron;

    class Layer;

    class OutputLayer;

    namespace act {}

    namespace make {
        struct NeuronOptions;
        struct LayerOptions;
    }

    using vd = std::vector<double>;
    using vn = std::vector<Neuron>;
}

#endif //FRUIT_CLASSIFIER_WASM_NN_H
