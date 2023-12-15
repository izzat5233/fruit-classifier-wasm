//
// Created by Izzat on 12/15/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_IO_UTILITY_H
#define FRUIT_CLASSIFIER_WASM_IO_UTILITY_H

#include <sstream>
#include <emscripten.h>
#include "emscripten_browser_file.h"
#include <nn.h>

#define UPLOAD_HANDLER_PARAMETERS \
const std::string &filename, const std::string &mime_type, std::string_view buffer, void *callback_data

void processCsvData(UPLOAD_HANDLER_PARAMETERS) {
    auto data = static_cast<nn::vvd_t *>(callback_data);
    data->clear();
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
            data->push_back(row);
        }
    }
}

#endif //FRUIT_CLASSIFIER_WASM_IO_UTILITY_H
