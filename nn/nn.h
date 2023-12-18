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
     * In case of 1-neuron size layer (Binary Classification) it uses Sigmoid activation function.
     */
    class OutputLayer;

    /**
     * Represents a neural network, which is a collection of layers.
     * Uses gradient descent training algorithm.
     */
    class Network;

    /**
     * The Module class encapsulates a neural network and
     * manages its training, testing, and regularization processes.
     * It also manages normalizing and de-normalizing with minmax function.
     */
    class Module;

    /*
     * Activation Functions Namespace
     */
    namespace act {
        /**
         * Base struct for activation functions in a neural network.
         * This struct defines the interface for activation functions and their derivatives,
         * allowing for polymorphic use of different activation functions within the network.
         */
        struct Function;
    }

    /**
     * Different types used frequently in nn namespace
     */
    inline namespace type {
        using ui_t = unsigned short int;
        using vi_t = std::vector<ui_t>;
        using vd_t = std::vector<double>;
        using vn_t = std::vector<Neuron>;
        using vl_t = std::vector<HiddenLayer>;
        using vf_t = std::vector<act::Function>;
        using fdd_t = double (*)(double);
        using vpd_t = std::vector<std::pair<double, double>>;
        using vvd_t = std::vector<std::vector<double>>;
        using vvvd_t = std::vector<vvd_t>;
    }

    namespace act {
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

    /**
     * Loss Functions Namespace
     */
    namespace loss {
        /**
         * The general type for a loss function.
         */
        using function_t = double (*)(const vd_t &, const vd_t &);

        /**
         * Calculates the Sum Square Error.
         * @param desired Desired output values
         * @param actual Actual output values
         * @return The Sum Square Error
         */
        double sse(const vd_t &desired, const vd_t &actual);

        /**
         * Calculates the Mean Square Error.
         * @param desired Desired output values
         * @param actual Actual output values
         * @return The Mean Square Error
         */
        double mse(const vd_t &desired, const vd_t &actual);
    }

    /**
     * Data Processing Namespace.
     * Contains Regularization & Normalization functions.
     */
    namespace process {
        /**
         * The general type for regularization functions.
         */
        using regularizer_t = double (*)(const vd_t &, double);

        /**
         * L1 Regularization.
         * Applies L1 regularization technique.
         * @param weights Vector of network weights
         * @param lambda Regularization coefficient
         * @return The L1 regularization term
         */
        double l1(const vd_t &weights, double lambda);

        /**
         * L2 Regularization.
         * Applies L2 regularization technique.
         * @param weights Vector of network weights
         * @param lambda Regularization coefficient
         * @return The L2 regularization term
         */
        double l2(const vd_t &weights, double lambda);

        /**
         * Min-Max Normalization.
         * Normalizes the data using Min-Max scaling.
         * @param data Vector of data points
         * @param minParam Minimum value parameter used in normalization.
         * @param maxParam Minimum value parameter used in normalization.
         * @return Normalized data vector
         */
        vd_t minmax(const vd_t &data, double minParam, double maxParam);

        /**
         * Inverse Min-Max Normalization (De-normalization).
         * Reverts the normalized data back to its original scale.
         * @param data Vector of normalized data points
         * @param originalMin The original minimum value of the data before normalization
         * @param originalMax The original maximum value of the data before normalization
         * @return Denormalized data vector
         */
        vd_t inverseMinmax(const vd_t &data, double originalMin, double originalMax);
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
        Neuron neuron(const ui_t &numInputs, double lowBound, double highBound);

        /**
         * Creates a vector of neurons with the specified options
         * which can be used to create a HiddenLayer or an OutputLayer.
         * Neurons weights and bias are given random values in the range:
         * [-numNeurons / rangeFactor, +numNeurons / rangeFactor].
         * Use this function only if specific network functionality is needed.
         *
         * @param numInputs Number of inputs for the neurons.
         * @param numNeurons Number of neurons for the layer.
         * @param rangeFactor Random values range factor. Default is 2.4.
         * @return A vector of properly configured neurons.
         */
        vn_t layer(const ui_t &numInputs, const ui_t &numNeurons, double rangeFactor = 2.4);

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
         * @param lossFunction Loss function used for backward propagation and error calculation.
         * @return A Network object configured as per the provided options.
         */
        Network network(const vi_t &dimensions, const vf_t &functions, loss::function_t lossFunction);

        /**
         * Creates a Network with the specified options.
         * It's recommended to use this method for creating neural networks.
         *
         * @param dimensions Dimensions of the neural network.
         * First value represents the number of inputs for the network.
         * Mid values represents the number of neurons for each hidden layer.
         * Last value represents the number of neurons for the output layer.
         * @param function Activation function used for all hidden layers in the network.
         * @param lossFunction Loss function used for backward propagation and error calculation.
         * @return A Network object configured as per the provided options.
         */
        Network network(const vi_t &dimensions, const act::Function &functions, loss::function_t lossFunction);
    }
}

#endif //FRUIT_CLASSIFIER_WASM_NN_H
