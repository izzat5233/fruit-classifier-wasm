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
    /**
     * Class representing a single neuron in a neural network.
     * This class encapsulates the neuron's weights and bias,
     * and provides functionality for computing the neuron's output.
     */
    class Neuron;

    /**
     * Represents a generic layer in a neural network. This class serves as a base for hidden & output layers.
     * And provides common attributes and methods needed for any neural network layer.
     */
    class Layer;

    /**
     * Represents a hidden layer in a neural network. A layer is a collection of neurons that
     * processes inputs and produces outputs based on the defined activation function.
     */
    class HiddenLayer;

    /**
     * Represents the output layer of the neural network. A collection of neurons that
     * processes inputs and produces outputs. Uses Softmax as an activation function.
     */
    class OutputLayer;

    /**
     * Represents a neural network, which is a collection of layers.
     */
    class Network;

    /**
     * Different types used frequently in nn namespace
     */
    inline namespace type {
        using size_t = unsigned char;
        using vd_t = std::vector<double>;
        using vn_t = std::vector<Neuron>;
        using vl_t = std::vector<HiddenLayer>;
        using vvd_t = std::vector<std::vector<double>>;
        using fdd_t = double (*)(double);
    }

    /**
     * The Factory SubNamespace
     */
    namespace make {
        /**
         * Struct to encapsulate options for creating a Neuron.
         */
        struct NeuronOptions;
        /**
         * Struct to encapsulate options for creating a Layer.
         */
        struct LayerOptions;
        /**
         * Struct to encapsulate options for creating a Network.
         */
        struct NetworkOptions;

        /**
         * Creates a Neuron with the specified options.
         *
         * @param options The NeuronOptions struct containing configuration for the neuron.
         * @return A Neuron object configured as per the provided options.
         */
        Neuron neuron(NeuronOptions options);

        /**
         * Creates a Layer with the specified options.
         * Use this core layer it to create a HiddenLayer or an OutputLayer.
         *
         * @param options The LayerOptions struct containing configuration for the layer.
         * @return A Layer object configured as per the provided options.
         */
        Layer layer(LayerOptions options);

        /**
         * Creates a Network with the specified options.
         *
         * @param options The NetworkOptions struct containing configuration for the layer.
         * @return A Network object configured as per the provided options.
         */
        Network network(NetworkOptions options);
    }

    /*
     * Activation Functions Namespace
     */
    namespace act {
        /**
         * Abstract base struct for activation functions in a neural network.
         * This struct defines the interface for activation functions and their derivatives,
         * allowing for polymorphic use of different activation functions within the network.
         */
        struct Function {
            /**
             * The activation function.
             * A function from x to y.
             */
            fdd_t fun;
            /**
             * The derivative of the activation function.
             * A function from y to y'.
             */
            fdd_t der;
        };

        extern Function step;
        extern Function sign;
        extern Function linear;
        extern Function relu;
        extern Function sigmoid;
        extern Function tanh;

        /**
         * Special activation function used for output layers
         * The softmax function converts a vector of real values into a probability distribution.
         *
         * @param t Vector of all input values
         * @return Vector of all output values
         */
        vd_t softmax(const vd_t &t);
    }
}

#endif //FRUIT_CLASSIFIER_WASM_NN_H
