//
// Created by Izzat on 12/17/2023.
//

#include "module.h"

#include <algorithm>
#include <cassert>

using namespace nn;

Module::Module()
        : network(), trainInput(), trainOutput(), testInput(), testOutput() {}

Module::Module(nn::Network network)
        : network(network), trainInput(), trainOutput(), testInput(), testOutput() {}

void Module::setNetwork(Network newNetwork) {
    this->network.emplace(std::move(newNetwork));
}

vvvd_t Module::getWeights() const {
    vvvd_t res;
    for (std::size_t i = 0; i < network->getSize(); ++i) {
        vvd_t weights;
        for (const Neuron &neuron: network->get(i)) {
            weights.push_back(neuron);
        }
        res.push_back(weights);
    }
    return res;
}

vvd_t Module::getBiases() const {
    vvd_t res;
    for (std::size_t i = 0; i < network->getSize(); ++i) {
        vd_t biases;
        for (const Neuron &neuron: network->get(i)) {
            biases.push_back(neuron.getBias());
        }
        res.push_back(biases);
    }
    return res;
}

void Module::setLearningRate(double learningRate) {
    this->alpha = learningRate;
}

double Module::getLearningRate() const {
    return alpha;
}

void Module::NormalizedData::set(const vvd_t &data) {
    minMax.clear();
    minMax.reserve(data.size());
    for (const vd_t &line: data) {
        double minVal = *std::min_element(line.begin(), line.end());
        double maxVal = *std::max_element(line.begin(), line.end());
        minMax.emplace_back(minVal, maxVal);
    }
    normalized = normalize(data);
}

vvd_t Module::NormalizedData::get() const {
    return denormalize(normalized);
}

const vvd_t &Module::NormalizedData::use() const {
    return normalized;
}

vvd_t Module::NormalizedData::normalize(const nn::vvd_t &original) const {
    vvd_t norm;
    norm.reserve(original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        norm.push_back(process::minmax(original[i], minMax[i].first, minMax[i].second));
    }
    return norm;
}

vvd_t Module::NormalizedData::denormalize(const vvd_t &processed) const {
    vvd_t original;
    original.reserve(processed.size());
    for (std::size_t i = 0; i < processed.size(); ++i) {
        original.push_back(process::inverseMinmax(processed[i], minMax[i].first, minMax[i].second));
    }
    return original;
}

void Module::setTrainInput(const vvd_t &data) {
    trainInput.set(data);
}

[[nodiscard]] vvd_t Module::getTrainInput() const {
    return trainInput.get();
}

void Module::setTrainOutput(const vvd_t &data) {
    trainOutput.set(data);
}

[[nodiscard]] vvd_t Module::getTrainOutput() const {
    return trainOutput.get();
}

void Module::setTestInput(const vvd_t &data) {
    testInput = data;
}

[[nodiscard]] vvd_t Module::getTestInput() const {
    return testInput;
}

void Module::setTestOutput(const vvd_t &data) {
    testOutput = data;
}

[[nodiscard]] vvd_t Module::getTestOutput() const {
    return testOutput;
}

double Module::train() {
    const vvd_t &inputs = trainInput.use();
    const vvd_t &outputs = trainOutput.use();
    assert(inputs.size() == outputs.size());

    double sum = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        sum += network->train(inputs[i], outputs[i], alpha);
    }
    return sum / (double) inputs.size();
}

double Module::test() const {
    const vvd_t &inputs = trainInput.normalize(testInput);
    const vvd_t &outputs = trainOutput.normalize(testOutput);
    assert(inputs.size() == outputs.size());

    double sum = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        sum += network->test(inputs[i], outputs[i]);
    }
    return sum / (double) inputs.size();
}

vd_t Module::train(std::size_t epochs) {
    vd_t errors(epochs);
    for (std::size_t i = 0; i < epochs; ++i) { errors[i] = train(); }
    return errors;
}

vd_t Module::test(std::size_t epochs) const {
    vd_t errors(epochs);
    for (std::size_t i = 0; i < epochs; ++i) { errors[i] = test(); }
    return errors;
}

vpd_t Module::trainAndTest(std::size_t epochs) {
    vpd_t errors(epochs);
    for (std::size_t i = 0; i < epochs; ++i) {
        errors[i].first = train();
        errors[i].second = test();
    }
    return errors;
}

vvd_t Module::predict() const {
    return predict(testInput);
}

vvd_t Module::predict(const vvd_t &inputData) const {
    const vvd_t &normalized = trainInput.normalize(inputData);
    vvd_t processed(normalized.size());
    for (std::size_t i = 0; i < processed.size(); ++i) {
        processed[i] = network->predict(normalized[i]);
    }
    return trainOutput.denormalize(processed);
}