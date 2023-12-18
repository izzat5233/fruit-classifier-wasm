//
// Created by Izzat on 11/27/2023.
//

#include <algorithm>
#include <cassert>
#include "network.h"

using namespace nn;

Network::Network(vl_t layers, OutputLayer outputLayer, loss::function_t lossFunction)
        : size(layers.size() + 1), layers(std::move(layers)),
          outputLayer(std::move(outputLayer)), lossFunction(lossFunction) {
    assert(!this->layers.empty());
    assert([this] {
        auto i = this->layers.cbegin();
        for (i = std::next(i); i != this->layers.cend(); i = std::next(i)) {
            if (i->cbegin()->size() != std::prev(i)->size()) { return false; }
        }
        return this->outputLayer.cbegin()->size() == this->layers.crbegin()->size();
    }());
}

std::size_t Network::getSize() const {
    return size;
}

Layer &Network::get(std::size_t index) {
    if (index == layers.size()) { return outputLayer; }
    return layers[index];
}

const Layer &Network::get(std::size_t index) const {
    if (index == layers.size()) { return outputLayer; }
    return layers[index];
}

Layer &Network::rget(std::size_t index) {
    if (index == 0) { return outputLayer; }
    return layers[size - 1 - index];
}

const Layer &Network::rget(std::size_t index) const {
    if (index == 0) { return outputLayer; }
    return layers[size - 1 - index];
}

vd_t Network::predict(const vd_t &input) const {
    auto res = input;
    for (std::size_t i = 0; i < size; ++i) { res = get(i).activate(res); }
    return res;
}

vd_t Network::forwardPropagate(const vd_t &input) {
    auto res = input;
    for (std::size_t i = 0; i < size; ++i) { res = get(i).activateAndCache(res); }
    return res;
}

void Network::backwardPropagate(const vd_t &desired) {
    auto res = desired;
    for (std::size_t i = 0; i < size; ++i) {
        rget(i).calculateGradientsAndCash(res);
        res = rget(i).propagateErrorBackward();
    }
}

double Network::train(const vd_t &input, const vd_t &output, double alpha) {
    vd_t res = forwardPropagate(input);
    backwardPropagate(output);

    for (std::size_t i = 0; i < size; ++i) {
        auto &layer = get(i);
        for (std::size_t j = 0; j < layer.size(); ++j) {
            auto &neuron = layer[j];
            auto &y = (i > 0 ? get(i - 1).getOutputCash() : input);
            auto e = layer.getGradientCash()[j];
            neuron.adjust(y, e, alpha);
        }
    }

    return lossFunction(res, output);
}

double Network::test(const vd_t &input, const vd_t &output) const {
    vd_t res = predict(input);
    return lossFunction(res, output);
}