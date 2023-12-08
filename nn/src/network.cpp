//
// Created by Izzat on 11/27/2023.
//

#include <algorithm>
#include "network.h"
#include "../../util/debug.h"

using namespace nn;

Network::Network(vl_t layers, OutputLayer outputLayer, double alpha)
        : size(layers.size() + 1), layers(std::move(layers)), outputLayer(std::move(outputLayer)), alpha(alpha) {
    ASSERT(!this->layers.empty())
    ASSERT([this] {
        for (auto i = std::next(this->layers.begin()); i != this->layers.end(); ++i) {
            if ((*i)[0].size() != std::prev(i)->size()) { return false; }
        }
        return this->outputLayer[0].size() == this->layers.rbegin()->size();
    }())
    PRINT("Network created with " << size << " layers and alpha " << alpha)
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
        res = rget(i).propagateErrorBackward(rget(i).calculateGradientsAndCash(res));
    }
}

void Network::propagate(const vd_t &input, const vd_t &desired) {
    forwardPropagate(input);
    PRINT("Forward propagation done")
    backwardPropagate(desired);
    PRINT("Backward propagation done")
}

double Network::train(const vd_t &input, const vd_t &output) {
    propagate(input, output);
    double sse = util::sse(output, rget(0).output_cash);
    PRINT("Propagation error: " << sse)

    for (std::size_t i = 0; i < size; ++i) {
        auto &layer = get(i);
        for (std::size_t j = 0; j < layer.size(); ++j) {
            auto &neuron = layer[j];
            auto &y = (i > 0 ? get(i - 1).output_cash : input);
            auto e = layer.gradient_cash[j];
            neuron.adjust(y, e, alpha);
        }
    }

    return sse;
}

void Network::train(const vpvd_t &data, std::size_t epochsLimit, double errorThreshold) {
    for (std::size_t i = 0; i < epochsLimit; ++i) {
        PRINT("Epoch: " << i + 1)
        double worstError = errorThreshold - 1;
        for (const auto &p: data) {
            worstError = std::max(worstError, train(p.first, p.second));
        }
        if (worstError < errorThreshold) { break; }
    }
}
