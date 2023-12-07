//
// Created by Izzat on 12/6/2023.
//

#include <gtest/gtest.h>
#include <layer.h>
#include <hidden_layer.h>
#include <output_layer.h>

class LayerTest : public ::testing::Test {
protected:
    const double epsilon;

    const nn::Neuron nl1, nl2;
    const nn::Layer l0;

    const nn::Neuron nh1, nh2;
    const nn::Function fh;
    const nn::HiddenLayer h0;

    const nn::Neuron no1, no2, no3;
    const nn::OutputLayer o0;

    nn::Layer layer;
    nn::HiddenLayer hiddenLayer;
    nn::OutputLayer outputLayer;

    LayerTest() :
            epsilon(1e-9),

            nl1({0.1, 0.2}, -0.1),
            nl2({-0.2, 0.1}, 0.2),
            l0({nl1, nl2}),

            nh1({-0.5, 0.2, 0.1}, 0.3),
            nh2({0.05, 0.3, -0.05}, -0.2),
            fh(nn::act::tanh),
            h0(nn::Layer({nh1, nh2}), fh),

            no1({-0.5, 0.2}, 0.3),
            no2({0.05, 0.3}, -0.2),
            no3({0.1, 0.7}, -0.5),
            o0(nn::Layer({no1, no2, no3})),

            layer(l0),
            hiddenLayer(h0),
            outputLayer(o0) {}
};

TEST_F(LayerTest, Constructors) {
    EXPECT_EQ(layer, l0);
}

TEST_F(LayerTest, ProcessInputs) {
    nn::vd_t inputs = {0.5, -0.5};
    nn::vd_t expected_outputs = {-0.15, 0.05};
    nn::vd_t actual_outputs = layer.process(inputs);

    EXPECT_EQ(expected_outputs.size(), actual_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i) {
        EXPECT_NEAR(actual_outputs[i], expected_outputs[i], epsilon);
    }
}

TEST_F(LayerTest, BackPropagate) {
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

TEST_F(LayerTest, ActivateHiddenLayer) {
    nn::vd_t input = {3, -2, 0.8};
    nn::vd_t expected = {fh.fun(nh1.process(input)), fh.fun(nh2.process(input))};
    nn::vd_t actual = hiddenLayer.activate(input);

    EXPECT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], actual[i], epsilon);
    }
}

TEST_F(LayerTest, ActivateOutputLayer) {
    nn::vd_t input = {0.2, -0.1};
    nn::vd_t expected = nn::act::softmax({no1.process(input), no2.process(input), no3.process(input)});
    nn::vd_t actual = outputLayer.activate(input);

    EXPECT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], actual[i], epsilon);
    }
}