//
// Created by Izzat on 11/26/2023.
//

#include "../nn.h"
#include "../../util/debug.h"

#include <valarray>

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
        auto exps = std::exp(x);
        vd_t outputs = exps / exps.sum();
        PRINT_ITER("Softmax activated outputs:", outputs)
        return outputs;
    }
}