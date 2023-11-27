//
// Created by Izzat on 11/27/2023.
//

#include <numeric>
#include <algorithm>
#include "../nn/network.h"
#include "../util/debug.h"

using namespace nn;

Network::Network(vl_t layers, OutputLayer outputLayer, double alpha)
        : layers(std::move(layers)), outputLayer(std::move(outputLayer)), alpha(alpha),
          size(this->layers.size() + 1), y_cash(size), e_cash(size) {
    ASSERT(!this->layers.empty())
    ASSERT([this] {
        for (auto i = std::next(this->layers.begin()); i != this->layers.end(); ++i) {
            if ((*i)[0].size() != std::prev(i)->size()) { return false; }
        }
        return this->outputLayer[0].size() == this->layers.rbegin()->size();
    }())
    PRINT("Network created with " << size << " layers and alpha " << alpha)
}

void Network::forwardPropagate(const vd_t &input) {
    auto it = y_cash.begin();
    auto acc = std::accumulate(layers.begin(), layers.end(), input, [&it](const auto &acc, const auto &i) {
        (*it) = i.activate(acc);
        return *(it++);
    });
    (*it) = outputLayer.activate(acc);
}

void Network::backwardPropagate(const vd_t &output) {
    // Output layer
    e_cash[size - 1].resize(y_cash[size - 1].size());
    std::transform(output.begin(), output.end(), y_cash[size - 1].begin(), e_cash[size - 1].begin(), std::minus<>());

    // First hidden layer
    e_cash[size - 2] = outputLayer.backPropagate(e_cash[size - 1], layers[size - 2]);
    if (size <= 2) { return; }

    // Rest of layers
    for (std::size_t j = 0; j < size - 2; ++j) {
        auto i = size - 3 - j;
        e_cash[i] = layers[i + 1].backPropagate(e_cash[i + 1], layers[i]);
    }
}

void Network::propagate(const vd_t &input, const vd_t &output) {
    forwardPropagate(input);
    PRINT("Forward propagation done")
    backwardPropagate(output);
    PRINT("Backward propagation done")
}

void Network::train(const vd_t &input, const vd_t &output) {
    propagate(input, output);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        for (std::size_t j = 0; j < layers[i].size(); ++j) {
            auto &neuron = layers[i][j];
            auto &y = (i > 0 ? y_cash[i - 1] : input);
            auto e = e_cash[i][j];
            neuron.adjust(y, e, alpha);
        }
    }
    for (std::size_t j = 0; j < outputLayer.size(); ++j) {
        auto &neuron = outputLayer[j];
        auto &y = y_cash[size - 2];
        auto e = e_cash[size - 1][j];
        neuron.adjust(y, e, alpha);
    }
}

void Network::train(const std::vector<std::pair<vd_t, vd_t>> &data) {
    for (const auto &p: data) { train(p.first, p.second); }
}

vd_t Network::predict(const vd_t &input) const {
    auto acc = std::accumulate(layers.begin(), layers.end(), input, [](const auto &acc, const auto &i) {
        return i.activate(acc);
    });
    return outputLayer.activate(acc);
}
