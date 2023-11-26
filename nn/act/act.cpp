//
// Created by Izzat on 11/26/2023.
//

#include "../nn.h"
#include "../../util/debug.h"

#include <valarray>
#include <numeric>

namespace nn::act {
    Function step{
            [](double x) -> double { return x >= 0 ? 1 : 0; },
            [](double y) -> double { return 0; }
    };

    Function sign{
            [](double x) -> double { return x >= 0 ? 1 : -1; },
            [](double y) -> double { return 0; }
    };

    Function linear{
            [](double x) -> double { return x; },
            [](double y) -> double { return 1; }
    };

    Function relu{
            [](double x) -> double { return x >= 0 ? x : 0; },
            [](double y) -> double { return y >= 0 ? 1 : 0; }
    };

    Function sigmoid{
            [](double x) -> double { return 1 / (1 + std::exp(-x)); },
            [](double y) -> double { return y * (1 - y); }
    };

    Function tanh{
            [](double x) -> double { return 2 / (1 + std::exp(-2 * x)) - 1; },
            [](double y) -> double { return 1 - y * y; }
    };

    vd_t softmax(const vd_t &x) {
        auto sum = std::accumulate(x.begin(), x.end(), 0.0, [](auto t, auto i) { return t + std::exp(i); });
        vd_t outputs(x);
        std::transform(outputs.begin(), outputs.end(), outputs.begin(), [sum](auto i) { return std::exp(i) / sum; });
        PRINT_ITER("Softmax activated outputs:", outputs)
        return outputs;
    }
}