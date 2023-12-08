//
// Created by Izzat on 12/8/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_GLOBALS_H
#define FRUIT_CLASSIFIER_WASM_GLOBALS_H

#include <gtest/gtest.h>

#ifndef EPSILON
#define EPSILON 1e-9
#endif

#define EXPECT_ALL_NEAR(val1, val2, abs_error) { \
    EXPECT_EQ(val1.size(), val2.size()); \
    for (std::size_t $i = 0; $i < val1.size(); ++$i) { \
    EXPECT_NEAR(val1[$i], val2[$i], abs_error); \
    }\
};

#endif //FRUIT_CLASSIFIER_WASM_GLOBALS_H
