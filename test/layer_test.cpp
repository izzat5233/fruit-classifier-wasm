//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <layer.h>
#include <hidden_layer.h>

class LayerTest : public ::testing::Test {
protected:
    const double epsilon;
    const nn::Neuron nl1, nl2;
    const nn::Layer l0;
    const nn::Neuron nh1, nh2;
    const nn::HiddenLayer h0;

    nn::Layer layer;

    LayerTest() :
            epsilon(1e-9),
            nl1({0.1, 0.2}, -0.1),
            nl2({-0.2, 0.1}, 0.2),
            l0({nl1, nl2}),
            nh1({-0.5, 0.2, 0.1}, 0.3),
            nh2({0.05, 0.3, -0.05}, -0.2),
            h0(nn::Layer({nh1, nh2}), {nullptr, nullptr}),
            layer(l0) {}
};

TEST_F(LayerTest, Constructors) {
    EXPECT_EQ(layer, l0);
}

TEST_F(LayerTest, ProcessInputsCorrectly) {
    nn::vd_t inputs = {0.5, -0.5};
    nn::vd_t expected_outputs = {-0.15, 0.05};
    nn::vd_t actual_outputs = layer.process(inputs);

    EXPECT_EQ(expected_outputs.size(), actual_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i) {
        EXPECT_NEAR(actual_outputs[i], expected_outputs[i], epsilon);
    }
}

TEST_F(LayerTest, BackPropagateCalculatesGradientsCorrectly) {
    nn::vd_t gs = {0.4, -0.3};
    nn::vd_t expected_gradients = {
            gs[0] * nl1[0] + gs[1] * nl2[0],
            gs[0] * nl1[1] + gs[1] * nl2[1],
    };
    nn::vd_t actual_gradients = layer.backPropagate(gs, h0);

    EXPECT_EQ(expected_gradients.size(), actual_gradients.size());
    for (std::size_t i = 0; i < expected_gradients.size(); ++i) {
        EXPECT_NEAR(actual_gradients[i], expected_gradients[i], epsilon);
    }
}