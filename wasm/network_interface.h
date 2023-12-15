//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
#define FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H

#include <optional>
#include <network.h>

#include <emscripten.h>
#include <emscripten/bind.h>
#include "io_interface.h"

class NetworkController {
private:
    nn::vi_t dimensions;
    nn::act::Function function{};
    double alpha{};

    std::optional<nn::Network> network;

public:
    void setDimensions(const nn::vi_t &networkDimensions) {
        this->dimensions = networkDimensions;
        if (network.has_value()) { build(); }
    }

    void setActivationFunction(const std::string &activationFunction) {
        if (activationFunction == "sigmoid") {
            function = nn::act::sigmoid;
        } else if (activationFunction == "tanh") {
            function = nn::act::tanh;
        } else {
            function = nn::act::relu;
        }
        if (network.has_value()) { build(); }
    }

    void setLearningRate(double learningRate) {
        this->alpha = learningRate;
        if (network.has_value()) { network->setAlpha(learningRate); }
    }

    void build() {
        network.emplace(nn::make::network(dimensions, function, alpha));
    }

    void prepareTrainingData() {
        IOController::inputTrainingData(dimensions.front(), dimensions.back());
    }

    void prepareTestingData() {
        IOController::inputTestingData(dimensions.front(), dimensions.back());
    }

    nn::vd_t trainFor(std::size_t epochs) {
        nn::vd_t errors(epochs);
        for (std::size_t i = 0; i < epochs; ++i) {
            errors[i] = network->train(IOController::getTrainingDataCash(), nn::loss::sse);
        }
        return errors;
    }
};

EMSCRIPTEN_BINDINGS(my_module) {
    using namespace emscripten;

    register_vector<int>("VectorInt");
    register_vector<nn::ui_t>("VectorUInt");
    register_vector<double>("VectorNum");

    class_<NetworkController>("Network")
            .constructor<>()
            .function("setDimensions", &NetworkController::setDimensions)
            .function("setActivationFunction", &NetworkController::setActivationFunction)
            .function("setLearningRate", &NetworkController::setLearningRate)
            .function("build", &NetworkController::build)
            .function("prepareTrainingData", &NetworkController::prepareTrainingData)
            .function("prepareTestingData", &NetworkController::prepareTestingData)
            .function("trainFor", &NetworkController::trainFor);
}

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
