//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
#define FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H

#include <optional>
#include <network.h>

#include <emscripten.h>
#include <emscripten/bind.h>
#include "io_utility.h"

class NetworkController {
private:
    nn::vi_t dimensions;
    nn::act::Function actFunction{};
    nn::loss::function_t lossFunction{};
    double alpha{};

    nn::vvd_t *inputTrainingData = new nn::vvd_t();
    nn::vvd_t *outputTrainingData = new nn::vvd_t();
    nn::vvd_t *inputTestingData = new nn::vvd_t();
    nn::vvd_t *outputTestingData = new nn::vvd_t();

    std::optional<nn::vpvd_t> trainingData;
    std::optional<nn::vpvd_t> testingData;
    std::optional<nn::Network> network;

    static nn::vpvd_t concatenateData(const nn::vvd_t &inputs, const nn::vvd_t &outputs) {
        nn::vpvd_t data;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            data.emplace_back(inputs[i], outputs[i]);
        }
        return data;
    }

public:
    ~NetworkController() {
        delete inputTrainingData;
        delete outputTrainingData;
        delete inputTestingData;
        delete outputTestingData;
    }

    void setDimensions(const nn::vi_t &networkDimensions) {
        this->dimensions = networkDimensions;
        if (network.has_value()) { build(); }
    }

    void setLearningRate(double learningRate) {
        this->alpha = learningRate;
        if (network.has_value()) { network->setAlpha(learningRate); }
    }

    void setActivationFunction(const std::string &function) {
        if (function == "sigmoid") {
            actFunction = nn::act::sigmoid;
        } else if (function == "tanh") {
            actFunction = nn::act::tanh;
        } else {
            actFunction = nn::act::relu;
        }
        if (network.has_value()) { build(); }
    }

    void setLossFunction(const std::string &function) {
        if (function == "mse") {
            lossFunction = nn::loss::mse;
        } else {
            lossFunction = nn::loss::sse;
        }
    }

    void build() {
        network.emplace(nn::make::network(dimensions, actFunction, alpha));
    }

    void promptInputTrainingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, inputTrainingData);
        if (inputTrainingData->size() == outputTrainingData->size()) { prepareTrainingData(); }
    }

    void promptOutputTrainingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outputTrainingData);
        if (inputTrainingData->size() == outputTrainingData->size()) { prepareTrainingData(); }
    }

    void promptInputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, inputTestingData);
        if (inputTestingData->size() == outputTestingData->size()) { prepareTestingData(); }
    }

    void promptOutputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outputTestingData);
        if (inputTestingData->size() == outputTestingData->size()) { prepareTestingData(); }
    }

    void prepareTrainingData() {
        trainingData.emplace(concatenateData(*inputTrainingData, *outputTrainingData));
    }

    void prepareTestingData() {
        testingData.emplace(concatenateData(*inputTestingData, *outputTestingData));
    }

    nn::vd_t trainFor(std::size_t epochs) {
        nn::vd_t errors(epochs);
        for (std::size_t i = 0; i < epochs; ++i) {
            errors[i] = network->train(trainingData.value(), lossFunction);
        }
        return errors;
    }

    nn::vvd_t trainAndTestFor(std::size_t epochs) {
        nn::vvd_t errors(epochs, nn::vd_t(2));
        for (std::size_t i = 0; i < epochs; ++i) {
            errors[i][0] = network->train(trainingData.value(), lossFunction);
            errors[i][1] = testingDataError();
        }
        return errors;
    }

    [[nodiscard]] nn::vvd_t predictTestingOutputs() const {
        nn::vvd_t res;
        for (auto &in: *inputTestingData) {
            res.push_back(network->predict(in));
        }
        return res;
    }

    [[nodiscard]] double testingDataError() const {
        double error = LONG_MAX;
        nn::vvd_t output = predictTestingOutputs();
        for (std::size_t i = 0; i < output.size(); ++i) {
            error = std::min(error, lossFunction(output[i], (*outputTestingData)[i]));
        }
        return error;
    }
};

EMSCRIPTEN_BINDINGS(my_module) {
    using namespace emscripten;

    register_vector<int>("VecInt");
    register_vector<nn::ui_t>("VecUInt");
    register_vector<double>("VecNum");
    register_vector<nn::vvd_t>("VecVecNum");


    class_<NetworkController>("Network")
            .constructor<>()
            .function("setDimensions", &NetworkController::setDimensions)
            .function("setLearningRate", &NetworkController::setLearningRate)
            .function("setActivationFunction", &NetworkController::setActivationFunction)
            .function("setLossFunction", &NetworkController::setLossFunction)
            .function("build", &NetworkController::build)
            .function("promptInputTrainingData", &NetworkController::promptInputTrainingData)
            .function("promptOutputTrainingData", &NetworkController::promptOutputTrainingData)
            .function("promptInputTestingData", &NetworkController::promptInputTestingData)
            .function("promptOutputTestingData", &NetworkController::promptOutputTestingData)
            .function("prepareTrainingData", &NetworkController::prepareTrainingData)
            .function("prepareTestingData", &NetworkController::prepareTestingData)
            .function("trainFor", &NetworkController::trainFor)
            .function("trainAndTestFor", &NetworkController::trainAndTestFor)
            .function("predictTestingOutputs", &NetworkController::predictTestingOutputs)
            .function("testingDataError", &NetworkController::testingDataError);
}

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
