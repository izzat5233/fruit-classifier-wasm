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
    nn::vi_t dimensions = {2, 3, 2};
    nn::act::Function actFunction = nn::act::relu;
    nn::loss::function_t lossFunction = nn::loss::sse;
    double alpha = 0.1;

    nn::Module module;
    nn::vvd_t predictedCash;

    FileData *inTrainFile = new FileData([this] {
        EM_ASM_ARGS({ onInputTrainingFileSet(UTF8ToString($0), $1) },
                    inTrainFile->getFilename().c_str(), inTrainFile->getData().size());
    });
    FileData *outTrainFile = new FileData([this] {
        EM_ASM_ARGS({ onOutputTrainingFileSet(UTF8ToString($0), $1) },
                    outTrainFile->getFilename().c_str(), outTrainFile->getData().size());
    });
    FileData *inTestFile = new FileData([this] {
        EM_ASM_ARGS({ onInputTestingFileSet(UTF8ToString($0), $1) },
                    inTestFile->getFilename().c_str(), inTestFile->getData().size());
    });
    FileData *outTestFile = new FileData([this] {
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
        module.setNetwork(nn::make::network(dimensions, actFunction, lossFunction));
        module.setLearningRate(alpha);
    }

    void setDimensions(const nn::vi_t &networkDimensions) {
        this->dimensions = networkDimensions;
        build();
    }

    void setLearningRate(double learningRate) {
        this->alpha = learningRate;
        module.setLearningRate(alpha);
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

    void setInputTrainingData(const std::string &filename, const nn::vvd_t &data) {
        inTrainFile->setFilename(filename);
        inTrainFile->setData(data);
    }

    void promptOutputTrainingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outTrainFile);
    }

    void setOutputTrainingData(const std::string &filename, const nn::vvd_t &data) {
        outTrainFile->setFilename(filename);
        outTrainFile->setData(data);
    }

    void promptInputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, inTestFile);
    }

    void setInputTestingData(const std::string &filename, const nn::vvd_t &data) {
        inTestFile->setFilename(filename);
        inTestFile->setData(data);
    }

    void promptOutputTestingData() {
        emscripten_browser_file::upload(".csv,.txt", processCsvData, outTestFile);
    }

    void setOutputTestingData(const std::string &filename, const nn::vvd_t &data) {
        outTestFile->setFilename(filename);
        outTestFile->setData(data);
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

    nn::vvd_t predictTestingOutputs() {
        return predictedCash = module.predict();
    }

    nn::vvd_t getPredictedOutputs() {
        return predictedCash;
    }

    void downloadPredictedOutputs() {
        std::string data = prepareCsvData(predictedCash);
        emscripten_browser_file::download("predicted.csv", "text/csv", data.c_str(), data.size());
    }

    nn::vvvd_t getWeights() {
        return module.getWeights();
    }

    nn::vvd_t getBiases() {
        return module.getBiases();
    }

    void downloadNetwork() {
        std::string data = prepareNetworkData(getWeights(), getBiases());
        emscripten_browser_file::download("network.json", "json", data.c_str(), data.size());
    }

    nn::vvd_t getTrainIn() {
        return module.getTrainInput();
    }

    nn::vvd_t getTrainInPreview() {
        return inTrainFile->getPreview();
    }

    void downloadTrainIn() {
        std::string data = prepareCsvData(module.getTrainInput());
        emscripten_browser_file::download("trainIn.csv", "text/csv", data.c_str(), data.size());
    }

    nn::vvd_t getTrainOut() {
        return module.getTrainOutput();
    }

    nn::vvd_t getTrainOutPreview() {
        return outTrainFile->getPreview();
    }

    void downloadTrainOut() {
        std::string data = prepareCsvData(module.getTrainOutput());
        emscripten_browser_file::download("trainOut.csv", "text/csv", data.c_str(), data.size());
    }

    nn::vvd_t getTestIn() {
        return module.getTestInput();
    }

    nn::vvd_t getTestInPreview() {
        return inTestFile->getPreview();
    }

    void downloadTestIn() {
        std::string data = prepareCsvData(module.getTestInput());
        emscripten_browser_file::download("testIn.csv", "text/csv", data.c_str(), data.size());
    }

    nn::vvd_t getTestOut() {
        return module.getTestOutput();
    }

    nn::vvd_t getTestOutPreview() {
        return outTestFile->getPreview();
    }

    void downloadTestOut() {
        std::string data = prepareCsvData(module.getTestOutput());
        emscripten_browser_file::download("testOut.csv", "text/csv", data.c_str(), data.size());
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
            .function("setInputTrainingData", &NetworkController::setInputTrainingData)
            .function("promptOutputTrainingData", &NetworkController::promptOutputTrainingData)
            .function("setOutputTrainingData", &NetworkController::setOutputTrainingData)
            .function("promptInputTestingData", &NetworkController::promptInputTestingData)
            .function("setInputTestingData", &NetworkController::setInputTestingData)
            .function("promptOutputTestingData", &NetworkController::promptOutputTestingData)
            .function("setOutputTestingData", &NetworkController::setOutputTestingData)
            .function("prepareTrainingData", &NetworkController::prepareTrainingData)
            .function("prepareTestingData", &NetworkController::prepareTestingData)
            .function("trainFor", &NetworkController::trainFor)
            .function("trainAndTestFor", &NetworkController::trainAndTestFor)
            .function("predictTestingOutputs", &NetworkController::predictTestingOutputs)
            .function("getPredictedOutputs", &NetworkController::getPredictedOutputs)
            .function("getWeights", &NetworkController::getWeights)
            .function("getBiases", &NetworkController::getBiases)
            .function("downloadNetwork", &NetworkController::downloadNetwork)
            .function("downloadPredictedOutputs", &NetworkController::downloadPredictedOutputs)
            .function("getTrainIn", &NetworkController::getTrainIn)
            .function("getTrainInPreview", &NetworkController::getTrainInPreview)
            .function("downloadTrainIn", &NetworkController::downloadTrainIn)
            .function("getTrainOut", &NetworkController::getTrainOut)
            .function("getTrainOutPreview", &NetworkController::getTrainOutPreview)
            .function("downloadTrainOut", &NetworkController::downloadTrainOut)
            .function("getTestIn", &NetworkController::getTestIn)
            .function("getTestInPreview", &NetworkController::getTestInPreview)
            .function("downloadTestIn", &NetworkController::downloadTestIn)
            .function("getTestOut", &NetworkController::getTestOut)
            .function("getTestOutPreview", &NetworkController::getTestOutPreview)
            .function("downloadTestOut", &NetworkController::downloadTestOut);
}

#endif //FRUIT_CLASSIFIER_WASM_NETWORK_INTERFACE_H
