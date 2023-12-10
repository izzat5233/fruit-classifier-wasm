//
// Created by Izzat on 12/7/2023.
//

#include <gtest/gtest.h>
#include <network.h>

#include "globals.h"

class NetworkTest : public ::testing::Test {
protected:
    const double alpha;
    const nn::Neuron n11, n12, n21, n22, n23, n31, n32;
    const nn::act::Function f1, f2;
    const nn::HiddenLayer l1, l2;
    const nn::OutputLayer l3;

    nn::Network network;

    NetworkTest() :
            alpha(0.3),
            n11({-0.1, 0.2}, 0.1),
            n12({0.2, -0.1}, -0.2),
            n21({-0.05, -0.2}, -0.3),
            n22({-0.05, -0.3}, 0.2),
            n23({-0.1, -0.7}, 0.5),
            n31({0.5, -0.2, -0.1}, -0.3),
            n32({-0.05, -0.3, 0.4}, 0.2),
            f1(nn::act::sigmoid),
            f2(nn::act::tanh),
            l1({n11, n12}, f1),
            l2({n21, n22, n23}, f2),
            l3({n31, n32}),
            network({l1, l2}, l3, alpha) {}
};

TEST_F(NetworkTest, ForwardPropagation) {
    nn::vd_t input = {1, 0};
    nn::vvd_t expected = {
            {0.5,            0.5},
            {-0.4011342849,  0.0249947929, 0.0996679946},
            {0.316920916177, 0.683079083822}
    };
    network.forwardPropagate(input);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_ALL_NEAR(network.get(i).getOutputCash(), expected[i], EPSILON)
    }
}

TEST_F(NetworkTest, BackwardPropagationDoesntBreak) {
    nn::vd_t input = {1, 0};
    network.forwardPropagate(input);
    nn::vd_t desired = {0, 1};
    network.backwardPropagate(desired);
}

TEST_F(NetworkTest, SimpleTrainingTest) {
    nn::vd_t input = {1, 0};
    nn::vd_t output = {0, 1};
    nn::vd_t expected = {0.316920916177, 0.683079083822};
    nn::vd_t actual = network.train(input, output);
    EXPECT_ALL_NEAR(actual, expected, EPSILON);
}