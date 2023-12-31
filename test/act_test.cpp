//
// Created by Izzat on 12/7/2023.
//

#include <gtest/gtest.h>
#include <nn.h>

#define EPSILON 1e-10

#include "globals.h"

TEST(ActTest, ReluFunction) {
    auto f = nn::act::relu.fun;
    EXPECT_EQ(f(-1), 0);
    EXPECT_EQ(f(-0.0001), 0);
    EXPECT_EQ(f(0), 0);
    EXPECT_EQ(f(0.00001), 0.00001);
    EXPECT_EQ(f(1), 1);
}

TEST(ActTest, ReluDerivative) {
    auto [f, f1] = nn::act::relu;
    EXPECT_EQ(f1(f(-1)), 0);
    EXPECT_EQ(f1(f(-0.001)), 0);
    EXPECT_TRUE(f1(f(0)) == 0 || f1(f(0)) == 1);
    EXPECT_EQ(f1(f(0.001)), 1);
    EXPECT_EQ(f1(f(1)), 1);
}

TEST(ActTest, SigmoidFunction) {
    auto f = nn::act::sigmoid.fun;
    EXPECT_NEAR(f(-1), 0.26894142137, EPSILON);
    EXPECT_NEAR(f(-0.001), 0.49975000002083, EPSILON);
    EXPECT_NEAR(f(0), 0.5, EPSILON);
    EXPECT_NEAR(f(0.001), 0.50024999997917, EPSILON);
    EXPECT_NEAR(f(1), 0.73105857863, EPSILON);
}

TEST(ActTest, SigmoidDerivative) {
    auto [f, f1] = nn::act::sigmoid;
    EXPECT_NEAR(f1(f(-1)), 0.19661193324, EPSILON);
    EXPECT_NEAR(f1(f(-0.001)), 0.2499999375, EPSILON);
    EXPECT_NEAR(f1(f(0)), 0.25, EPSILON);
    EXPECT_NEAR(f1(f(0.001)), 0.2499999375, EPSILON);
    EXPECT_NEAR(f1(f(1)), 0.19661193324, EPSILON);
}

TEST(ActTest, TanhFunction) {
    auto f = nn::act::tanh.fun;
    EXPECT_NEAR(f(-1), -0.7615941559, EPSILON);
    EXPECT_NEAR(f(-0.001), -0.0009999996, EPSILON);
    EXPECT_NEAR(f(0), 0, EPSILON);
    EXPECT_NEAR(f(0.001), 0.0009999996, EPSILON);
    EXPECT_NEAR(f(1), 0.7615941559, EPSILON);
}

TEST(ActTest, TanhDerivative) {
    auto [f, f1] = nn::act::tanh;
    EXPECT_NEAR(f1(f(-1)), 0.4199743416, EPSILON);
    EXPECT_NEAR(f1(f(-0.001)), 0.9999990000, EPSILON);
    EXPECT_NEAR(f1(f(0)), 1, EPSILON);
    EXPECT_NEAR(f1(f(0.001)), 0.9999990000, EPSILON);
    EXPECT_NEAR(f1(f(1)), 0.4199743416, EPSILON);
}

TEST(ActTest, Softmax) {
    auto test = [](const nn::vd_t &input, const nn::vd_t &expected) {
        auto actual = nn::act::softmax(input);
        EXPECT_ALL_NEAR(actual, expected, EPSILON)
    };
    test({-0.5}, {1});
    test({1, 3, 2}, {0.0900305732, 0.6652409558, 0.2447284711});
    test({-1, 0, 0.5}, {0.1219516523, 0.3314989604, 0.5465493873});
    test({-1, 0, 2, 4, 2}, {0.00520014, 0.0141354461, 0.1044476041, 0.7717692058, 0.1044476041});
}