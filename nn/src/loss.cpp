//
// Created by Izzat on 11/28/2023.
//

#include <valarray>
#include <cassert>
#include "nn.h"

using namespace nn;

double loss::sse(const vd_t &desired, const vd_t &actual) {
    assert(desired.size() == actual.size());
    auto acc = 0.0;
    for (auto i = desired.begin(), j = actual.begin(); i != desired.end(); ++i, ++j) {
        auto diff = (*i) - (*j);
        acc += diff * diff;
    }
    return acc;
}

double loss::mse(const nn::vd_t &desired, const nn::vd_t &actual) {
    auto n = static_cast<double>(desired.size());
    assert(n == desired.size());
    return sse(desired, actual) / n;
}