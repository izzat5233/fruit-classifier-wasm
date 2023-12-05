#include "util/debug.h"
#include "nn/network.h"
#include <chrono>
#include <iostream>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    auto network = nn::make::network({2, 2, 2}, nn::act::relu, 0.01);
    nn::vpvd_t data = {
            {{0, 0}, {1, 0}},
            {{0, 1}, {0, 1}},
            {{1, 0}, {0, 1}},
            {{1, 1}, {1, 0}}
    };
    network.train(data, 10000, 0.1);

    std::cout << "Training Done.\n";
    for (const auto &p: data) {
        PRINT_ITER("Input: ", p.first)
        auto res = network.predict(p.first);
        PRINT_ITER("Output: ", res)
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "duration: " << duration.count() / (long double) 1000.0 << "s\n";
    return 0;
}