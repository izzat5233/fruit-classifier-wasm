//
// Created by Izzat on 12/7/2023.
//

#include <gtest/gtest.h>
#include <nn.h>

constexpr auto epsilon = 1e-3;

TEST(UtilTest, SSE) {
    EXPECT_NEAR(nn::util::sse({1}, {3}), 4, epsilon);
    EXPECT_NEAR(nn::util::sse({3}, {1}), 4, epsilon);
    EXPECT_NEAR(nn::util::sse(
            {6, 7, 7, 8, 12, 14, 15, 16, 16, 19},
            {14, 15, 15, 17, 18, 18, 16, 14, 11, 8}
    ), 476, epsilon);
}

TEST(UtilTest, MSE) {
    EXPECT_NEAR(nn::util::sse({1}, {3}), 4, epsilon);
    EXPECT_NEAR(nn::util::mse({3}, {1}), 4, epsilon);
    EXPECT_NEAR(nn::util::mse({1, 2}, {3, 4}), 4, epsilon);
    EXPECT_NEAR(nn::util::mse(
            {6, 7, 7, 8, 12, 14, 15, 16, 16, 19},
            {14, 15, 15, 17, 18, 18, 16, 14, 11, 8}
    ), 47.6, epsilon);
}