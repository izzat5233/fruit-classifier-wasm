//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_IO_UTILITY_H
#define FRUIT_CLASSIFIER_WASM_IO_UTILITY_H

#include <sstream>
#include <utility>
#include <emscripten.h>
#include "emscripten_browser_file.h"
#include <nn.h>

#define UPLOAD_HANDLER_PARAMETERS \
const std::string &filename, const std::string &mime_type, std::string_view buffer, void *callback_data

class CSVFile {
    std::function<void(void)> onProcessDone;
    std::string filename;
    nn::vvd_t data;

    friend void processCsvData(UPLOAD_HANDLER_PARAMETERS);

public:
    explicit CSVFile(std::function<void(void)> onProcessDone) {
        this->onProcessDone = std::move(onProcessDone);
    }

    [[nodiscard]] const nn::vvd_t &getData() const {
        return data;
    }

    [[nodiscard]] const std::string &getFilename() const {
        return filename;
    }

    [[nodiscard]] nn::vvd_t getPreview() const {
        nn::vvd_t res;
        std::size_t inc = std::max((std::size_t) 1, data.size() / 10);
        for (std::size_t i = 0; i < data.size(); i += inc) { res.push_back(data[i]); }
        return res;
    }
};

void processCsvData(UPLOAD_HANDLER_PARAMETERS) {
    auto file = static_cast<CSVFile *>(callback_data);
    file->data.clear();
    file->filename = filename;

    std::stringstream ss((std::string(buffer)));
    std::string line;

    while (std::getline(ss, line)) {
        nn::vd_t row;
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            try {
                double value = std::stod(cell);
                row.push_back(value);
            } catch (const std::invalid_argument &ia) {
                EM_ASM_ARGS({ console.error("Invalid argument: " + $0) }, ia.what());
                row.clear();
                break;
            }
        }

        if (!row.empty()) {
            file->data.push_back(row);
        }
    }

    file->onProcessDone();
}

std::string prepareCsvData(const nn::vvd_t &data) {
    std::stringstream ss;
    for (const auto &row: data) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            ss << row[i];
            if (i < row.size() - 1) { ss << ","; }
        }
        ss << "\n";
    }
    return ss.str();
}

/**
 * Prepares the network's weights and biases data for download in a simple JSON-like format.
 * @param weights The weights of the neural network.
 * @param biases The biases of the neural network.
 * @return A string representation of the network data in JSON-like format.
 */
std::string prepareNetworkData(const vvvd_t &weights, const vvd_t &biases) {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"weights\": [\n";
    for (size_t i = 0; i < weights.size(); ++i) {
        oss << "    [\n";
        for (size_t j = 0; j < weights[i].size(); ++j) {
            oss << "      [";
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                oss << weights[i][j][k];
                if (k < weights[i][j].size() - 1) oss << ", ";
            }
            oss << "]";
            if (j < weights[i].size() - 1) oss << ",";
            oss << "\n";
        }
        oss << "    ]";
        if (i < weights.size() - 1) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";

    oss << "  \"biases\": [\n";
    for (size_t i = 0; i < biases.size(); ++i) {
        oss << "    [";
        for (size_t j = 0; j < biases[i].size(); ++j) {
            oss << biases[i][j];
            if (j < biases[i].size() - 1) oss << ", ";
        }
        oss << "]";
        if (i < biases.size() - 1) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";

    return oss.str();
}

#endif //FRUIT_CLASSIFIER_WASM_IO_UTILITY_H
