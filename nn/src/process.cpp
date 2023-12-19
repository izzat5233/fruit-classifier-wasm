//
// Created by Izzat on 12/17/2023.
//

#include "nn.h"
#include <cassert>

using namespace nn;

vd_t process::minmax(const vd_t &data, const vpd_t &minMaxParams) {
    assert(data.size() == minMaxParams.size());
    vd_t normalized(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto [minParam, maxParam] = minMaxParams[i];
        if (minParam == maxParam) { normalized[i] = 0.5; }
        else normalized[i] = (data[i] - minParam) / (maxParam - minParam);
    }
    return normalized;
}

vd_t process::inverseMinmax(const vd_t &data, const vpd_t &minMaxParams) {
    assert(data.size() == minMaxParams.size());
    vd_t denormalized(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto [minParam, maxParam] = minMaxParams[i];
        denormalized[i] = data[i] * (maxParam - minParam) + minParam;
    }
    return denormalized;
}