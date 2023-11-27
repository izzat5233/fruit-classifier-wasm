//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "../../util/debug.h"

#include <algorithm>

using namespace nn;

Layer::Layer(vn_t neurons) : vn_t(std::move(neurons)) {
    ASSERT([this] {
        auto n = (*this)[0].size();
        return std::all_of(begin(), end(), [n](auto &i) { return i.size() == n; });
    }())
    PRINT("Layer created with " << size() << " neurons")
}

HiddenLayer::HiddenLayer(Layer layer, act::Function function)
        : Layer(std::move(layer)), function(function) {
    PRINT("And its a hidden layer")
}

OutputLayer::OutputLayer(Layer layer) : Layer(std::move(layer)) {
    PRINT("And its an output layer")
}

vd_t Layer::process(const vd_t &inputs) const {
    vd_t outputs(size());
    std::transform(begin(), end(), outputs.begin(), [&inputs](auto &n) { return n.process(inputs); });
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To raw outputs:", outputs)
    return outputs;
}

vd_t Layer::backPropagate(const vd_t &gradients, const HiddenLayer &previous) const {
    auto n = gradients.size();
    auto m = previous.size();
    ASSERT(n == size())
    ASSERT(m == (*this)[0].size())

    vd_t e(m);
    auto j = gradients.begin();
    for (auto i = begin(); i != end(); ++i, ++j) {
        auto &neuron = *i;
        auto s = *j;
        std::transform(e.begin(), e.end(), neuron.begin(), e.begin(), [s](auto acc, auto w) {
            return acc + w * s;
        });
    }

    std::transform(e.begin(), e.end(), e.begin(), previous.function.der);
    return e;
}

vd_t HiddenLayer::activate(const vd_t &inputs) const {
    vd_t outputs(Layer::process(inputs));
    std::transform(outputs.begin(), outputs.end(), outputs.begin(), this->function.fun);
    PRINT_ITER("Then to activated outputs:", outputs)
    return outputs;
}

vd_t OutputLayer::activate(const vd_t &inputs) const {
    return act::softmax(Layer::process(inputs));
}