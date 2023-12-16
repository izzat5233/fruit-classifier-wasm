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
    nn::vi_t dimensions = {2, 3, 2};
    nn::act::Function actFunction = nn::act::relu;
    nn::loss::function_t lossFunction = nn::loss::sse;
    double alpha = 0.1;

    CSVFile *inTrainFile = new CSVFile([this] {
        EM_ASM_ARGS({ onInputTrainingFileSet(UTF8ToString($0), $1) },
                    inTrainFile->getFilename().c_str(), inTrainFile->getData().size());
    });
    CSVFile *outTrainFile = new CSVFile([this] {
        EM_ASM_ARGS({ onOutputTrainingFileSet(UTF8ToString($0), $1) },
                    outTrainFile->getFilename().c_str(), outTrainFile->getData().size());
    });
    CSVFile *inTestFile = new CSVFile([this] {
        EM_ASM_ARGS({ onInputTestingFileSet(UTF8ToString($0), $1) },
                    inTestFile->getFilename().c_str(), inTestFile->getData().size());
    });
    CSVFile *outTestFile = new CSVFile([this] {
        EM_ASM_ARGS({ onOutputTestingFileSet(UTF8ToString($0), $1) },
                    outTestFile->getFilename().c_str(), outTestFile->getData().size());
    });

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
    NetworkController() {
        build();
    }

    ~NetworkController() {
        delete inTrainFile;
        delete outTrainFile;
        delete inTestFile;
        delete outTestFile;
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
        emscripten_browser_file::upload(".csv,.txt", processCsvData, inTrainFile);
    }

    void promptOutputTrainingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outTrainFile);
    }

    void promptInputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, inTestFile);
    }

    void promptOutputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outTestFile);
    }

    void prepareTrainingData() {
        trainingData.emplace(concatenateData(inTrainFile->getData(), outTrainFile->getData()));
        EM_ASM_ARGS({ onTrainingDataPrepared($0) }, trainingData->size());
    }

    void prepareTestingData() {
        testingData.emplace(concatenateData(inTestFile->getData(), outTestFile->getData()));
        EM_ASM_ARGS({ onTestingDataPrepared($0) }, testingData->size());
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
        for (const auto &in: inTestFile->getData()) {
            res.push_back(network->predict(in));
        }
        return res;
    }

    [[nodiscard]] double testingDataError() const {
        double avgError = 0;
        nn::vvd_t output = predictTestingOutputs();
        const nn::vvd_t &actual = outTestFile->getData();
        for (std::size_t i = 0; i < output.size(); ++i) {
            double error = lossFunction(output[i], actual[i]);
            avgError += error / output.size();
        }
        return avgError;
    }

    [[nodiscard]] std::vector<nn::vvd_t> getWeights() const {
        std::vector<nn::vvd_t> weights;
        weights.reserve(network->getSize());
        for (std::size_t i = 0; i < network->getSize(); ++i) {
            nn::vvd_t layerWeights;
            layerWeights.reserve(network->get(i).size());
            for (auto &neuron: network->get(i)) {
                nn::vd_t neuronWeights;
                neuronWeights.reserve(neuron.size());
                for (auto &weight: neuron) {
                    neuronWeights.push_back(weight);
                }
                layerWeights.push_back(neuronWeights);
            }
            weights.push_back(layerWeights);
        }
        return weights;
    }
};

EMSCRIPTEN_BINDINGS(my_module) {
    using namespace emscripten;

    register_vector<int>("VecInt");
    register_vector<nn::ui_t>("VecUInt");

    register_vector<double>("VecNum");
    register_vector<nn::vd_t>("VecVecNum");
    register_vector<nn::vvd_t>("VecVecVecNum");

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
            .function("testingDataError", &NetworkController::testingDataError)
            .function("getWeights", &NetworkController::getWeights);
}

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
