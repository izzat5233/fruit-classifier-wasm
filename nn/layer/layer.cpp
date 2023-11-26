//
// Created by Izzat on 11/24/2023.
//

#include "layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "../neuron/neuron.h"
#include "../make.h"
#include "../../util/debug.h"

#include <utility>
#include <numeric>
#include <valarray>

using namespace nn;

Layer::Layer(vn_t neurons) : neurons(std::move(neurons)) {
    ASSERT(std::all_of(this->neurons.begin(), this->neurons.end(), [this](auto &n) {
        return n.size() == this->neurons[0].size();
    }))
    PRINT("Layer created with " << this->neurons.size() << " neurons")
}

HiddenLayer::HiddenLayer(Layer layer, fdd_t actFun)
        : Layer(std::move(layer)), actFun(actFun) {
    PRINT("And its a hidden layer")
}

OutputLayer::OutputLayer(Layer layer) : Layer(std::move(layer)) {
    PRINT("And its an output layer")
}

nn::size_t Layer::size() const {
    return static_cast<size_t>(neurons.size());
}

vd_t Layer::process(const vd_t &inputs) const {
    vd_t outputs(neurons.size());
    std::transform(neurons.begin(), neurons.end(), outputs.begin(), [&inputs](auto &n) { return n.process(inputs); });
    PRINT_ITER("Layer processed inputs:", inputs)
    PRINT_ITER("To raw outputs:", outputs)
    return outputs;
}

vd_t Layer::calculateErrors(const vd_t &gradients, const HiddenLayer &previousLayer) const {
    auto n = gradients.size();
    auto m = previousLayer.size();
    ASSERT(n == neurons.size())
    ASSERT(m == neurons[0].size())
    vd_t errors(m);
    for (int i = 0; i < n; ++i) {
        auto &neuron = neurons[i];
        auto sigma = gradients[i];
        for (size_t j = 0; j < m; ++j) {
            errors[j] += neuron.weights[j] * sigma;
        }
    }
    return errors;
}

vd_t HiddenLayer::activate(const vd_t &inputs) const {
    vd_t outputs(Layer::process(inputs));
    std::transform(outputs.begin(), outputs.end(), outputs.begin(), this->actFun);
    PRINT_ITER("Then to activated outputs:", outputs)
    return outputs;
}

vd_t act::softmax(const vd_t &x) {
    auto total = std::accumulate(x.begin(), x.end(), 0.0, [](auto acc, auto i) { return acc + exp(i); });
    vd_t outputs(x);
    std::transform(outputs.begin(), outputs.end(), outputs.begin(), [total](auto i) { return exp(i) / total; });
    PRINT_ITER("Softmax activated outputs:", outputs)
    return outputs;
}

vd_t OutputLayer::activate(const vd_t &inputs) const {
    return act::softmax(Layer::process(inputs));
}

Layer make::layer(nn::make::LayerOptions options) {
    auto [numInputs, numNeurons] = options;
    make::NeuronOptions neuronOptions(numInputs, -numNeurons / 2.4, numNeurons / 2.4);
    vn_t neurons;
    neurons.reserve(numNeurons);
    for (int i = 0; i < numNeurons; ++i) { neurons.push_back(make::neuron(neuronOptions)); }
    return Layer(neurons);
}