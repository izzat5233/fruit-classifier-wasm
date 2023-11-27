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
     * Uses gradient descent training algorithm.
     */
    class Network;

    /**
     * Base struct for activation functions in a neural network.
     * This struct defines the interface for activation functions and their derivatives,
     * allowing for polymorphic use of different activation functions within the network.
     */
    struct Function;

    /**
     * Different types used frequently in nn namespace
     */
    inline namespace type {
        using ui_t = unsigned short int;
        using vi_t = std::vector<ui_t>;
        using vd_t = std::vector<double>;
        using vn_t = std::vector<Neuron>;
        using vl_t = std::vector<HiddenLayer>;
        using vf_t = std::vector<Function>;
        using vvd_t = std::vector<std::vector<double>>;
        using fdd_t = double (*)(double);
    }

    struct Function {
        /**
         * The activation function.
         * A function from x to y.
         */
        const fdd_t &fun;
        /**
         * The derivative of the activation function.
         * A function from y to y'.
         */
        const fdd_t &der;
    };

    /*
     * Activation Functions Namespace
     */
    namespace act {
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

    /**
     * The Factory Namespace
     */
    namespace make {
        /**
         * Creates a Neuron with the specified options.
         * Weights and bias are given random values between the given boundaries.
         * Use this function only if specific weights are needed.
         *
         * @param numInputs Number of inputs for the neuron.
         * @param lowBound Lower bound for random weight initialization.
         * @param highBound Upper bound for random weight initialization.
         * @return A Neuron object configured as per the provided options.
         */
        Neuron neuron(const ui_t &numInputs, const double &lowBound, const double &highBound);

        /**
         * Creates a Layer with the specified options
         * which can be used to create a HiddenLayer or an OutputLayer.
         * Neurons weights and bias are given random values in the range:
         * [-numNeurons / 2.4, +numNeurons / 2.4].
         * Use this function only if specific network functionality is needed.
         *
         * @param numInputs Number of inputs for the neurons.
         * @param numNeurons Number of neurons for the layer.
         * @return A Layer object configured as per the provided options.
         */
        Layer layer(const ui_t &numInputs, const ui_t &numNeurons);

        /**
         * Creates a Network with the specified options.
         * It's recommended to use this method for creating neural networks.
         *
         * @param dimensions Dimensions of the neural network.
         * First value represents the number of inputs for the network.
         * Mid values represents the number of neurons for each hidden layer.
         * Last value represents the number of neurons for the output layer.
         * @param functions Activation functions used for each hidden layer in the neural network.
         * For N dimensions there should be N - 2 functions.
         * @param alpha Initial learning rate used for the neural network.
         * @return A Network object configured as per the provided options.
         */
        Network network(const vi_t &dimensions, const vf_t &functions, double alpha);

        /**
         * Creates a Network with the specified options.
         * It's recommended to use this method for creating neural networks.
         *
         * @param dimensions Dimensions of the neural network.
         * First value represents the number of inputs for the network.
         * Mid values represents the number of neurons for each hidden layer.
         * Last value represents the number of neurons for the output layer.
         * @param function Activation function used for all hidden layers in the network.
         * @param alpha Initial learning rate used for the neural network.
         * @return A Network object configured as per the provided options.
         */
        Network network(const vi_t &dimensions, const Function &function, double alpha);
    }
}

#endif //FRUIT_CLASSIFIER_WASM_NN_H
