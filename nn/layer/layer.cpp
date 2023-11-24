//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "output_layer.h"
#include "../../util/debug.h"

#include <utility>
#include <numeric>

using namespace nn;

Layer::Layer(vn neurons, act::type function)
        : neurons(std::move(neurons)), function(std::move(function)) {
    ASSERT(std::all_of(this->neurons.begin(), this->neurons.end(), [this](const auto &n) {
        return n.size() == this->neurons[0].size();
    }))
    PRINT("Layer created with " << this->neurons.size() << " neurons")
}

vd Layer::calculateOutputs(const vd &inputs) const {
    vd outputs(neurons.size());
    std::transform(neurons.begin(), neurons.end(), outputs.begin(), [&inputs](const auto &n) {
        return n.process(inputs);
    });
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To raw outputs:", outputs)
    return outputs;
}

vd Layer::process(const nn::vd &inputs) const {
    vd outputs(calculateOutputs(inputs));
    std::transform(outputs.begin(), outputs.end(), outputs.begin(), this->function);
    PRINT_ITER("Then activated them to:", outputs)
    return outputs;
}

Layer make::layer(make::LayerOptions options) {
    auto [n, o, f] = std::move(options);
    vn neurons;
    neurons.reserve(n);
    for (int i = 0; i < n; ++i) { neurons.push_back(make::neuron(o)); }
    return Layer(neurons, f);
}

vd act::softmax(const vd &x) {
    auto total = std::accumulate(x.begin(), x.end(), 0.0, [](auto acc, auto i) { return acc + exp(i); });
    vd outputs(x);
    std::transform(outputs.begin(), outputs.end(), outputs.begin(), [total](auto i) { return exp(i) / total; });
    PRINT_ITER("Softmax result:", outputs)
    return outputs;
}

OutputLayer::OutputLayer(vn neurons) : Layer(std::move(neurons), act::linear) {
    PRINT("And it was an output layer")
}

vd OutputLayer::process(const vd &inputs) const {
    return act::softmax(calculateOutputs(inputs));
}
