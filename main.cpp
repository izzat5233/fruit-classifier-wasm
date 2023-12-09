#include <network.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using std::cout;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::string;
using std::stringstream;

// Function to read data from a CSV file
nn::vpvd_t readData(const string &filename) {
    nn::vpvd_t data;
    ifstream file(filename);
    string line;

    while (std::getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        nn::vd_t inputs;
        nn::vd_t outputs;

        while (getline(lineStream, cell, ',')) {
            inputs.push_back(std::stod(cell));
        }

        double label = inputs.back();
        inputs.pop_back();
        outputs.push_back(label);
        data.emplace_back(inputs, outputs);
    }

    return data;
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    auto network = nn::make::network({4, 3, 1}, nn::act::relu, 0.01);

    nn::vpvd_t trainingData = readData("../datasets/sample/training.csv");
    network.train(trainingData, 1000, 0.1);
    cout << "Training Done.\n";

    nn::vpvd_t testingData = readData("../datasets/sample/testing.csv");
    ofstream resultFile("../datasets/sample/result.csv");

    for (const auto &p: testingData) {
        auto res = network.predict(p.first);
        for (auto i: p.first) { resultFile << i << ','; }
        for (auto r: res) { resultFile << r << ' '; }
        resultFile << '\n';
    }
    resultFile.close();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "duration: " << duration.count() / (long double) 1000.0 << "s\n";
    return 0;
}