//
// Created by Izzat on 12/17/2023.
//

#include "nn.h"
#include <algorithm>

using namespace nn;

vd_t process::minmax(const vd_t &data) {
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    if (minVal == maxVal) { return {(data.size(), 0.5)}; }

    vd_t normalized(data.size());
    std::transform(data.begin(), data.end(), normalized.begin(), [minVal, maxVal](double x) {
        return (x - minVal) / (maxVal - minVal);
    });
    return normalized;
}

vd_t process::inverse_minmax(const vd_t &data, double originalMin, double originalMax) {
    if (originalMin == originalMax) { return {(data.size(), originalMin)}; }
    vd_t denormalized(data.size());
    std::transform(data.begin(), data.end(), denormalized.begin(), [originalMin, originalMax](double x) {
        return x * (originalMax - originalMin) + originalMin;
    });
    return denormalized;
}