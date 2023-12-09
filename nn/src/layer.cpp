//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "../../util/debug.h"

#include <algorithm>
#include <utility>

using namespace nn;

Layer::Layer(vn_t neurons)
        : vn_t(std::move(neurons)), output_cash(size()), gradient_cash(size()) {
    ASSERT([this] {
        auto n = (*this)[0].size();
        return std::all_of(begin(), end(), [n](auto &i) { return i.size() == n; });
    }())
    PRINT("Layer created with " << size() << " neurons")
}

HiddenLayer::HiddenLayer(vn_t neurons, act::Function function)
        : Layer(std::move(neurons)), function(function) {
    PRINT("And its a hidden layer")
}

OutputLayer::OutputLayer(vn_t neurons) : Layer(std::move(neurons)) {
    PRINT("And its an output layer")
}

const vd_t &Layer::getOutputCash() const {
    return output_cash;
}

const vd_t &Layer::getGradientCash() const {
    return gradient_cash;
}

vd_t Layer::process(const vd_t &inputs) const {
    vd_t outputs(size());
    std::transform(begin(), end(), outputs.begin(), [&inputs](auto &n) { return n.process(inputs); });
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To raw outputs:", outputs)
    return outputs;
}

vd_t Layer::activateAndCache(const vd_t &inputs) {
    return output_cash = activate(inputs);
}

vd_t HiddenLayer::activate(const vd_t &inputs) const {
    vd_t output = Layer::process(inputs);
    std::transform(output.begin(), output.end(), output.begin(), this->function.fun);
    PRINT_ITER("Then to activated outputs:", outputs)
    return output;
}

vd_t OutputLayer::activate(const vd_t &inputs) const {
    return act::softmax(Layer::process(inputs));
}

vd_t Layer::propagateErrorBackward(const vd_t &gradients) const {
    ASSERT(gradients.size() == size())

    vd_t e((*this)[0].size());
    for (std::size_t i = 0; i < e.size(); ++i) {
        auto g = gradients.begin();
        for (auto n = begin(); n != end(); ++n, ++g) { e[i] += (*n)[i] * (*g); }
    }
    return e;
}

vd_t Layer::calculateGradientsAndCash(const vd_t &intermediateGradients) {
    return gradient_cash = calculateGradients(intermediateGradients);
}

vd_t HiddenLayer::calculateGradients(const vd_t &intermediateGradients) const {
    ASSERT(size() == intermediateGradients.size())
    vd_t gradients(size());
    for (std::size_t i = 0; i < size(); ++i) {
        gradients[i] = intermediateGradients[i] * function.der(output_cash[i]);
    }
    return gradients;
}

vd_t OutputLayer::calculateGradients(const vd_t &intermediateGradients) const {
    ASSERT(size() == intermediateGradients.size())
    vd_t gradients(size());
    for (std::size_t i = 0; i < size(); ++i) {
        gradients[i] = output_cash[i] - intermediateGradients[i];
    }
    return gradients;
}
