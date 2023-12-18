//
// Created by Izzat on 12/17/2023.
//

#include "nn.h"
#include <algorithm>

using namespace nn;

vd_t process::minmax(const vd_t &data, double minParam, double maxParam) {
    if (data.size() == 1) { return {data[0]}; } // turn off for single outputs
    if (minParam == maxParam) { return vd_t(data.size(), 0.5); }

    vd_t normalized(data.size());
    std::transform(data.begin(), data.end(), normalized.begin(), [minParam, maxParam](double x) {
        return (x - minParam) / (maxParam - minParam);
    });
    return normalized;
}

vd_t process::inverseMinmax(const vd_t &data, double originalMin, double originalMax) {
    if (data.size() == 1) { return {data[0]}; } // turn off for single outputs
    vd_t denormalized(data.size());
    std::transform(data.begin(), data.end(), denormalized.begin(), [originalMin, originalMax](double x) {
        return x * (originalMax - originalMin) + originalMin;
    });
    return denormalized;
}