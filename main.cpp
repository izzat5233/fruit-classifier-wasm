#include "nn/network/network.h"

#include <iostream>
#include <vector>

using std::cout, std::cin, std::vector;

int main() {
    auto network = nn::make::network({2, 3, 2}, nn::act::tanh, 1);
    vector<double> input = {1, 2}, output = {1, 0};
    network.train(input, output);
}
