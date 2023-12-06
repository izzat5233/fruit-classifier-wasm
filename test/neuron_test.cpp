//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <neuron.h>

class NeuronTest : public ::testing::Test {
protected:
    const double epsilon;
    const nn::Neuron n0;

    nn::Neuron neuron;

    NeuronTest() :
            epsilon(1e-12),
            n0({0.1, -0.2, 0.3}, -1),
            neuron(n0) {}
};

TEST_F(NeuronTest, Constructors) {
    EXPECT_EQ(neuron, n0);
    EXPECT_EQ(neuron.getBias(), n0.getBias());
}

TEST_F(NeuronTest, AdjustWeightsManually) {
    neuron.adjust({0.2, 0.1, -0.5}, 2);
    EXPECT_NEAR(neuron[0], n0[0] + 0.2, epsilon);
    EXPECT_NEAR(neuron[1], n0[1] + 0.1, epsilon);
    EXPECT_NEAR(neuron[2], n0[2] - 0.5, epsilon);
    EXPECT_NEAR(neuron.getBias(), n0.getBias() + 2, epsilon);
}

TEST_F(NeuronTest, AdjustWeightsWithGradient) {
    neuron.adjust({0.2, 0.1, -0.5}, 0.5, 0.1);
    auto factor = 0.5 * 0.1;
    EXPECT_NEAR(neuron[0], n0[0] + 0.2 * factor, epsilon);
    EXPECT_NEAR(neuron[1], n0[1] + 0.1 * factor, epsilon);
    EXPECT_NEAR(neuron[2], n0[2] + -0.5 * factor, epsilon);
    EXPECT_NEAR(neuron.getBias(), n0.getBias() + -1 * factor, epsilon);
}

TEST_F(NeuronTest, AdjustWeightsWithZeroGradient) {
    neuron.adjust({0.2, 0.1, -0.5}, 0, 0.1);
    EXPECT_NEAR(neuron[0], n0[0], epsilon);
    EXPECT_NEAR(neuron[1], n0[1], epsilon);
    EXPECT_NEAR(neuron[2], n0[2], epsilon);
    EXPECT_NEAR(neuron.getBias(), n0.getBias(), epsilon);
}

TEST_F(NeuronTest, AdjustWeightsWithZeroLearningRate) {
    neuron.adjust({0.2, 0.1, -0.5}, 0.5, 0);
    EXPECT_NEAR(neuron[0], n0[0], epsilon);
    EXPECT_NEAR(neuron[1], n0[1], epsilon);
    EXPECT_NEAR(neuron[2], n0[2], epsilon);
    EXPECT_NEAR(neuron.getBias(), n0.getBias(), epsilon);
}

TEST_F(NeuronTest, ProcessWeightedSum) {
    double result = neuron.process({-0.5, 0.125, 0.75});
    double actual = n0[0] * -0.5 + n0[1] * 0.125 + n0[2] * 0.75 + n0.getBias();
    EXPECT_NEAR(result, actual, epsilon);
}

TEST_F(NeuronTest, ProcessWithZerosInput) {
    double result = neuron.process({0, 0, 0});
    EXPECT_NEAR(result, n0.getBias(), epsilon);
}