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
            }
        }

        if (!row.empty()) {
            file->data.push_back(row);
        }
    }

    file->onProcessDone();
}

#endif //FRUIT_CLASSIFIER_WASM_IO_UTILITY_H
