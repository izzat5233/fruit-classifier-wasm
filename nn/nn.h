//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NN_H
#define FRUIT_CLASSIFIER_WASM_NN_H

#include <vector>
#include <functional>
#include <memory>

/**
 * Neural Network Namespace
 */
namespace nn {
    class Neuron;

    class Layer;

    class OutputLayer;

    class Network;

    namespace make {
        struct NeuronOptions;
        struct LayerOptions;
    }

    namespace act {
        struct Function;
        using ptr = std::unique_ptr<Function>;
    }

    inline namespace type {
        using vd_t = std::vector<double>;
        using vn_t = std::vector<Neuron>;
        using vl_t = std::vector<std::unique_ptr<Layer>>;
        using fd_t = std::function<double(double)>;
    }
}

#endif //FRUIT_CLASSIFIER_WASM_NN_H
