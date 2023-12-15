//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_IO_INTERFACE_H
#define FRUIT_CLASSIFIER_WASM_IO_INTERFACE_H

#include <sstream>
#include <emscripten.h>
#include "emscripten_browser_file.h"
#include <nn.h>

#define UPLOAD_HANDLER_PARAMETERS \
const std::string &filename, const std::string &mime_type, std::string_view buffer, void *callback_data
#define UPLOAD_HANDLER_ARGUMENTS \
filename, mime_type, buffer, callback_data

class IOController {
private:
    static nn::vvd_t inputCash;

    static nn::ui_t inputSizeCash;

    static nn::ui_t outputSizeCash;

    static nn::vpvd_t trainingDataCash;

    static nn::vpvd_t testingDataCash;

    static void cashCsvData(UPLOAD_HANDLER_PARAMETERS) {
        nn::vvd_t data;
        std::stringstream ss((std::string(buffer)));
        std::string line;

        while (std::getline(ss, line)) {
            std::vector<double> row;
            std::stringstream lineStream(line);
            std::string cell;

            bool add = true;
            while (std::getline(lineStream, cell, ',')) {
                try {
                    double value = std::stod(cell);
                    row.push_back(value);
                } catch (const std::invalid_argument &ia) {
                    EM_ASM_ARGS({ console.error("Invalid argument: " + $0) }, ia.what());
                    add = false;
                }
            }

            if (add && !row.empty()) {
                data.push_back(row);
            }
        }

        inputCash = data;
    }

    static void cashTrainingData(UPLOAD_HANDLER_PARAMETERS) {
        cashCsvData(UPLOAD_HANDLER_ARGUMENTS);
        trainingDataCash = convertToPairs();
    }

    static void cashTestingData(UPLOAD_HANDLER_PARAMETERS) {
        cashCsvData(UPLOAD_HANDLER_ARGUMENTS);
        testingDataCash = convertToPairs();
    }

    static nn::vpvd_t convertToPairs() {
        nn::vpvd_t cash(inputCash.size());
        for (std::size_t i = 0; i < inputCash.size(); ++i) {
            auto &[in, out] = cash[i];
            in.resize(inputSizeCash);
            out.resize(outputSizeCash);
            for (std::size_t j = 0; j < in.size(); ++j) { in[j] = inputCash[i][j]; }
            for (std::size_t j = 0; j < out.size(); ++j) { out[j] = inputCash[i][j + in.size()]; }
        }
        return cash;
    }

public:
    static void inputTrainingData(nn::ui_t inputSize, nn::ui_t outputSize) {
        inputSizeCash = inputSize;
        outputSizeCash = outputSize;
        emscripten_browser_file::upload(".csv,.txt", cashTrainingData);
    }

    static void inputTestingData(nn::ui_t inputSize, nn::ui_t outputSize) {
        inputSizeCash = inputSize;
        outputSizeCash = outputSize;
        emscripten_browser_file::upload(".csv,.txt", cashTestingData);
    }

    static const nn::vvd_t &getInputCash() {
        return inputCash;
    }

    static const nn::vpvd_t &getTrainingDataCash() {
        return trainingDataCash;
    }

    static const nn::vpvd_t &getTestingDataCash() {
        return testingDataCash;
    }
};

inline nn::vvd_t IOController::inputCash;
inline nn::ui_t IOController::inputSizeCash;
inline nn::ui_t IOController::outputSizeCash;
inline nn::vpvd_t IOController::trainingDataCash;
inline nn::vpvd_t IOController::testingDataCash;

#endif //FRUIT_CLASSIFIER_WASM_IO_INTERFACE_H
