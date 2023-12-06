//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <neuron.h>

constexpr auto epsilon = 1e-6;

static const nn::vd_t weights{0.1, -0.2, 0.3};
static const double bias = -1;

// Test fixture for Neuron
class NeuronTest : public ::testing::Test {
protected:
    NeuronTest() : neuron(weights, bias) {
    }

    ~NeuronTest() override = default;

    void SetUp() override {}

    void TearDown() override {}

    nn::Neuron neuron;
};

TEST_F(NeuronTest, DefaultConstructor) {
    EXPECT_NEAR(neuron[0], weights[0], epsilon);
    EXPECT_NEAR(neuron[1], weights[1], epsilon);
    EXPECT_NEAR(neuron[2], weights[2], epsilon);
    EXPECT_NEAR(neuron.getBias(), bias, epsilon);
}

TEST_F(NeuronTest, AdjustWeightsManually) {
    neuron.adjust({0.2, 0.1, -0.5}, 2);
    EXPECT_NEAR(neuron[0], weights[0] + 0.2, epsilon);
    EXPECT_NEAR(neuron[1], weights[1] + 0.1, epsilon);
    EXPECT_NEAR(neuron[2], weights[2] - 0.5, epsilon);
    EXPECT_NEAR(neuron.getBias(), bias + 2, epsilon);
}

TEST_F(NeuronTest, AdjustWeightsByGradient) {
    neuron.adjust({0.2, 0.1, -0.5}, 0.5, 0.1);
    auto factor = 0.5 * 0.1;
    EXPECT_NEAR(neuron[0], weights[0] + 0.2 * factor, epsilon);
    EXPECT_NEAR(neuron[1], weights[1] + 0.1 * factor, epsilon);
    EXPECT_NEAR(neuron[2], weights[2] + -0.5 * factor, epsilon);
    EXPECT_NEAR(neuron.getBias(), bias + -1 * factor, epsilon);
}

TEST_F(NeuronTest, ProcessWeightedSum) {
    double result = neuron.process({-0.5, 0.125, 0.75});
    double actual = weights[0] * -0.5 + weights[1] * 0.125 + weights[2] * 0.75 + bias;
    EXPECT_NEAR(result, actual, epsilon);
}