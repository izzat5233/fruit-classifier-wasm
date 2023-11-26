//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "../../util/debug.h"

#include <utility>
#include <numeric>
#include <valarray>

using namespace nn;

Layer::Layer(vn_t neurons) : vn_t(std::move(neurons)) {
    ASSERT(std::all_of(begin(), end(), [this](auto &n) { return n.size() == (*this)[0].size(); }))
    PRINT("Layer created with " << size() << " neurons")
}

HiddenLayer::HiddenLayer(Layer layer, fdd_t actFun)
        : Layer(std::move(layer)), actFun(actFun) {
    PRINT("And its a hidden layer")
}

OutputLayer::OutputLayer(Layer layer) : Layer(std::move(layer)) {
    PRINT("And its an output layer")
}

vd_t Layer::process(const vd_t &inputs) const {
    vd_t outputs(size());
    for (size_t i = 0; i < size(); ++i) { outputs[i] = (*this)[i].process(inputs); }
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To raw outputs:", outputs)
    return outputs;
}

vd_t Layer::calculateErrors(const vd_t &gradients, const HiddenLayer &previousLayer) const {
    auto n = gradients.size();
    auto m = previousLayer.size();
    ASSERT(n == size())
    ASSERT(m == (*this)[0].size())

    vd_t errors(m);
    for (int i = 0; i < n; ++i) {
        auto &neuron = (*this)[i];
        auto sigma = gradients[i];
        for (size_t j = 0; j < m; ++j) { errors[j] += neuron[j] * sigma; }
    }

    return errors;
}

vd_t HiddenLayer::activate(const vd_t &inputs) const {
    vd_t outputs(Layer::process(inputs).apply(this->actFun));
    PRINT_ITER("Then to activated outputs:", outputs)
    return outputs;
}

vd_t OutputLayer::activate(const vd_t &inputs) const {
    return act::softmax(Layer::process(inputs));
}