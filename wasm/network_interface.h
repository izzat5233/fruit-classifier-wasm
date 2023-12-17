//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
#define FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H

#include <optional>
#include <module.h>

#include <emscripten.h>
#include <emscripten/bind.h>
#include "io_utility.h"

class NetworkController {
private:
    nn::Module module;
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

    static nn::vvd_t pairToVector(const nn::vpd_t &data) {
        nn::vvd_t res;
        for (const auto &[i, j]: data) { res.push_back({i, j}); }
        return res;
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

    void build() {
        module.setNetwork(nn::make::network(dimensions, actFunction, lossFunction, alpha));
    }

    void setDimensions(const nn::vi_t &networkDimensions) {
        this->dimensions = networkDimensions;
        build();
    }

    void setLearningRate(double learningRate) {
        this->alpha = learningRate;
        build();
    }

    void setActivationFunction(const std::string &function) {
        if (function == "sigmoid") {
            actFunction = nn::act::sigmoid;
        } else if (function == "tanh") {
            actFunction = nn::act::tanh;
        } else {
            actFunction = nn::act::relu;
        }
        build();
    }

    void setLossFunction(const std::string &function) {
        if (function == "mse") {
            lossFunction = nn::loss::mse;
        } else {
            lossFunction = nn::loss::sse;
        }
        build();
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
        module.setTrainInput(inTrainFile->getData());
        module.setTrainOutput(outTrainFile->getData());
        EM_ASM_ARGS({ onTrainingDataPrepared($0) }, inTrainFile->getData().size());
    }

    void prepareTestingData() {
        module.setTestInput(inTestFile->getData());
        module.setTestOutput(outTestFile->getData());
        EM_ASM_ARGS({ onTestingDataPrepared($0) }, inTestFile->getData().size());
    }

    nn::vd_t trainFor(std::size_t epochs) {
        return module.train(epochs);
    }

    nn::vvd_t trainAndTestFor(std::size_t epochs) {
        return pairToVector(module.trainAndTest(epochs));
    }

    [[nodiscard]] nn::vvd_t predictTestingOutputs() const {
        return module.predict();
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
            .function("predictTestingOutputs", &NetworkController::predictTestingOutputs);
}

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
