//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "../../util/debug.h"

#include <utility>

using namespace nn;

Layer::Layer(vn neurons, act::type function)
        : neurons(std::move(neurons)), function(std::move(function)) {
    ASSERT(std::all_of(this->neurons.begin(), this->neurons.end(), [this](const auto &n) {
        return n.size() == this->neurons[0].size();
    }))
    PRINT("Layer created with " << this->neurons.size() << " neurons")
}

vd Layer::process(const nn::vd &inputs) const {
    vd output(neurons.size());
    std::transform(neurons.begin(), neurons.end(), output.begin(), [this, &inputs](const auto &n) {
        return this->function(n.process(inputs));
    });
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To outputs:", output)
    return output;
}

void Layer::setActivationFunction(const act::type &activationFunction) {
    this->function = activationFunction;
}

act::type Layer::getActivationFunction() const {
    return function;
}

Layer make::layer(make::LayerOptions options) {
    auto [n, o, f] = std::move(options);
    vn neurons;
    for (int i = 0; i < n; ++i) { neurons.push_back(make::neuron(o)); }
    return Layer(neurons, f);
}
