//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "hidden_layer.h"
#include "output_layer.h"

#include <algorithm>
#include <utility>
#include <cassert>

using namespace nn;

Layer::Layer(vn_t neurons) : vn_t(std::move(neurons)), output_cash(size()), gradient_cash(size()) {}

HiddenLayer::HiddenLayer(vn_t neurons, act::Function function) : Layer(std::move(neurons)), function(function) {}

OutputLayer::OutputLayer(vn_t neurons) : Layer(std::move(neurons)) {}

const vd_t &Layer::getOutputCash() const {
    return output_cash;
}

const vd_t &Layer::getGradientCash() const {
    return gradient_cash;
}

vd_t Layer::process(const vd_t &inputs) const {
    vd_t res(size());
    std::transform(begin(), end(), res.begin(), [&inputs](auto &n) { return n.process(inputs); });
    return res;
}

vd_t Layer::activateAndCache(const vd_t &inputs) {
    return output_cash = activate(inputs);
}

vd_t HiddenLayer::activate(const vd_t &inputs) const {
    vd_t res = Layer::process(inputs);
    std::transform(res.begin(), res.end(), res.begin(), this->function.fun);
    return res;
}

vd_t OutputLayer::activate(const vd_t &inputs) const {
    vd_t res = Layer::process(inputs);
    if (size() == 1) { return {act::sigmoid.fun(res[0])}; }
    return act::softmax(res);
}

vd_t Layer::propagateErrorBackward() const {
    vd_t e(begin()->size());
    for (std::size_t i = 0; i < e.size(); ++i) {
        auto g = gradient_cash.begin();
        for (auto n = begin(); n != end(); ++n, ++g) { e[i] += (*n)[i] * (*g); }
    }
    return e;
}

vd_t Layer::calculateGradientsAndCash(const vd_t &intermediateGradients) {
    return gradient_cash = calculateGradients(intermediateGradients);
}

vd_t HiddenLayer::calculateGradients(const vd_t &intermediateGradients) const {
    assert(size() == intermediateGradients.size());
    vd_t gradients(size());
    for (std::size_t i = 0; i < size(); ++i) {
        gradients[i] = intermediateGradients[i] * function.der(output_cash[i]);
    }
    return gradients;
}

vd_t OutputLayer::calculateGradients(const vd_t &intermediateGradients) const {
    assert(size() == intermediateGradients.size());
    vd_t gradients(size());
    for (std::size_t i = 0; i < size(); ++i) {
        gradients[i] = output_cash[i] - intermediateGradients[i];
    }
    return gradients;
}
