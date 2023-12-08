#include <network.h>
#include <chrono>
#include <iostream>

using std::cout;

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
    cout << "Training Done.\n";

    for (const auto &p: data) {
        cout << "Inputs: ";
        for (auto i: p.first) { cout << i << ' '; }
        cout << '\n';

        // Predict the output
        auto res = network.predict(p.first);

        cout << "Outputs: ";
        for (auto i: res) { cout << i << ' '; }
        cout << '\n';
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "duration: " << duration.count() / (long double) 1000.0 << "s\n";
    return 0;
}