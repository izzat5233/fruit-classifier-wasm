//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_ACT_H
#define FRUIT_CLASSIFIER_WASM_ACT_H

#include <functional>
#include <cmath>

/*
 * Activation Functions Namespace
 */
namespace nn::act {
    /**
     * Abstract base struct for activation functions in a neural network.
     * This struct defines the interface for activation functions and their derivatives,
     * allowing for polymorphic use of different activation functions within the network.
     */
    struct Function {
        /**
         * Virtual destructor to ensure proper cleanup of derived classes.
         */
        virtual ~Function() = default;

        /**
         * The activation function.
         * @param x input
         * @return output
         */
        [[nodiscard]] virtual double fun(double x) const noexcept = 0;

        /**
         * The derivative of the activation function.
         * @param x input
         * @return output
         */
        [[nodiscard]] virtual double der(double x) const noexcept = 0;
    };

    struct Step : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return x >= 0 ? 1 : 0; }

        [[nodiscard]] double der(double x) const noexcept override { return 0; }
    };

    struct Sign : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return x >= 0 ? 1 : -1; }

        [[nodiscard]] double der(double x) const noexcept override { return 0; }
    };

    struct Linear : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return x; }

        [[nodiscard]] double der(double x) const noexcept override { return 1; }
    };

    struct ReLU : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return x >= 0 ? x : 0; }

        [[nodiscard]] double der(double x) const noexcept override { return x >= 0 ? 1 : 0; }
    };

    struct Sigmoid : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return 1 / (1 + exp(-x)); }

        [[nodiscard]] double der(double x) const noexcept override { return fun(x) * (1 - fun(x)); }
    };

    struct Tanh : Function {
        [[nodiscard]] double fun(double x) const noexcept override { return 1 / (1 + exp(-x)); }

        [[nodiscard]] double der(double x) const noexcept override { return 1 - fun(x) * fun(x); }
    };
}

#endif //FRUIT_CLASSIFIER_WASM_ACT_H
