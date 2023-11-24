//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_ACTIVATION_H
#define FRUIT_CLASSIFIER_WASM_ACTIVATION_H

#include <functional>
#include <cmath>

/*
 * Activation Functions Namespace
 */
namespace act {
    using function = std::function<double(double)>;

    constexpr double step(double x) noexcept {
        return x >= 0 ? 1 : 0;
    }

    constexpr double sign(double x) noexcept {
        return x >= 0 ? 1 : -1;
    }

    constexpr double linear(double x) noexcept {
        return x;
    }

    constexpr double relu(double x) noexcept {
        return x >= 0 ? x : 0;
    }

    constexpr double sigmoid(double x) noexcept {
        return 1 / (1 + exp(-x));
    }

    constexpr double tanh(double x) noexcept {
        return 2 / (1 + exp(-2 * x)) - 1;
    }
}

#endif //FRUIT_CLASSIFIER_WASM_ACTIVATION_H
