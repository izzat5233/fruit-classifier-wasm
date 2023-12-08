//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <layer.h>
#include <hidden_layer.h>
#include <output_layer.h>

#include "globals.h"

class LayerTest : public ::testing::Test {
protected:
    const nn::Neuron nl1, nl2;
    const nn::HiddenLayer l0;

    const nn::Neuron nh1, nh2;
    const nn::Function fh;
    const nn::HiddenLayer h0;

    const nn::Neuron no1, no2, no3;
    const nn::OutputLayer o0;

    nn::HiddenLayer layer;
    nn::HiddenLayer hiddenLayer;
    nn::OutputLayer outputLayer;

    LayerTest() :
            nl1({0.1, 0.2}, -0.1),
            nl2({-0.2, 0.1}, 0.2),
            l0({nl1, nl2}, {}),

            nh1({-0.5, 0.2, 0.1}, 0.3),
            nh2({0.05, 0.3, -0.05}, -0.2),
            fh(nn::act::tanh),
            h0({nh1, nh2}, fh),

            no1({-0.5, 0.2}, 0.3),
            no2({0.05, 0.3}, -0.2),
            no3({0.1, 0.7}, -0.5),
            o0({no1, no2, no3}),

            layer(l0),
            hiddenLayer(h0),
            outputLayer(o0) {}
};

TEST_F(LayerTest, Constructors) {
    EXPECT_EQ(hiddenLayer, h0);
    EXPECT_EQ(outputLayer, o0);
}

TEST_F(LayerTest, ProcessInputs) {
    nn::vd_t input = {0.5, -0.5};
    nn::vd_t expected = {-0.15, 0.05};
    nn::vd_t actual = layer.process(input);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}

TEST_F(LayerTest, PreProcessGradients) {
    nn::vd_t gradients = {0.4, -0.3};
    nn::vd_t expected = {
            gradients[0] * nl1[0] + gradients[1] * nl2[0],
            gradients[0] * nl1[1] + gradients[1] * nl2[1],
    };
    nn::vd_t actual = layer.propagateErrorBackward(gradients);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}

TEST_F(LayerTest, ActivateHiddenLayer) {
    nn::vd_t input = {3, -2, 0.8};
    nn::vd_t expected = {fh.fun(nh1.process(input)), fh.fun(nh2.process(input))};
    nn::vd_t actual = hiddenLayer.activate(input);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}

TEST_F(LayerTest, ActivateOutputLayer) {
    nn::vd_t input = {0.2, -0.1};
    nn::vd_t expected = nn::act::softmax({no1.process(input), no2.process(input), no3.process(input)});
    nn::vd_t actual = outputLayer.activate(input);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}

TEST_F(LayerTest, CalculateGradientsForHiddenLayer) {
    nn::vd_t output = hiddenLayer.activateAndCache({3, -2, 0.8});
    nn::vd_t preGradients = {-0.2, 0.1};
    nn::vd_t expected = {
            preGradients[0] * fh.der(output[0]),
            preGradients[1] * fh.der(output[1])
    };
    nn::vd_t actual = hiddenLayer.calculateGradients(preGradients);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}

TEST_F(LayerTest, CalculateGradientsForOutputLayer) {
    nn::vd_t output = outputLayer.activateAndCache({0.2, -0.1});
    nn::vd_t desired = {-2, 0.8, 0.5};
    nn::vd_t expected = {
            output[0] - desired[0],
            output[1] - desired[1],
            output[2] - desired[2]
    };
    nn::vd_t actual = outputLayer.calculateGradients(desired);
    EXPECT_ALL_NEAR(expected, actual, EPSILON)
}