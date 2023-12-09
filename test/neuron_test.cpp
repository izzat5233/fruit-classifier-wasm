//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <neuron.h>

#define EPSILON 1e-12

#include "globals.h"

class NeuronTest : public ::testing::Test {
protected:
    const nn::Neuron n0;
    nn::Neuron neuron;

    NeuronTest() :
            n0({0.1, -0.2, 0.3}, -1),
            neuron(n0) {}
};

TEST_F(NeuronTest, Constructors) {
    EXPECT_EQ(neuron, n0);
    EXPECT_EQ(neuron.getBias(), n0.getBias());
}

TEST_F(NeuronTest, AdjustWeightsManually) {
    neuron.adjust({0.2, 0.1, -0.5}, 2);
    EXPECT_NEAR(neuron[0], n0[0] + 0.2, EPSILON);
    EXPECT_NEAR(neuron[1], n0[1] + 0.1, EPSILON);
    EXPECT_NEAR(neuron[2], n0[2] - 0.5, EPSILON);
    EXPECT_NEAR(neuron.getBias(), n0.getBias() + 2, EPSILON);
}

TEST_F(NeuronTest, AdjustWeightsWithGradient) {
    neuron.adjust({0.2, 0.1, -0.5}, 0.5, 0.1);
    auto factor = -1 * 0.5 * 0.1;
    EXPECT_NEAR(neuron[0], n0[0] + 0.2 * factor, EPSILON);
    EXPECT_NEAR(neuron[1], n0[1] + 0.1 * factor, EPSILON);
    EXPECT_NEAR(neuron[2], n0[2] + -0.5 * factor, EPSILON);
    EXPECT_NEAR(neuron.getBias(), n0.getBias() + factor, EPSILON);
}

TEST_F(NeuronTest, AdjustWeightsWithZeroGradient) {
    neuron.adjust({0.2, 0.1, -0.5}, 0, 0.1);
    EXPECT_NEAR(neuron[0], n0[0], EPSILON);
    EXPECT_NEAR(neuron[1], n0[1], EPSILON);
    EXPECT_NEAR(neuron[2], n0[2], EPSILON);
    EXPECT_NEAR(neuron.getBias(), n0.getBias(), EPSILON);
}

TEST_F(NeuronTest, AdjustWeightsWithZeroLearningRate) {
    neuron.adjust({0.2, 0.1, -0.5}, 0.5, 0);
    EXPECT_NEAR(neuron[0], n0[0], EPSILON);
    EXPECT_NEAR(neuron[1], n0[1], EPSILON);
    EXPECT_NEAR(neuron[2], n0[2], EPSILON);
    EXPECT_NEAR(neuron.getBias(), n0.getBias(), EPSILON);
}

TEST_F(NeuronTest, ProcessWeightedSum) {
    double result = neuron.process({-0.5, 0.125, 0.75});
    double actual = n0[0] * -0.5 + n0[1] * 0.125 + n0[2] * 0.75 + n0.getBias();
    EXPECT_NEAR(result, actual, EPSILON);
}

TEST_F(NeuronTest, ProcessWithZerosInput) {
    double result = neuron.process({0, 0, 0});
    EXPECT_NEAR(result, n0.getBias(), EPSILON);
}