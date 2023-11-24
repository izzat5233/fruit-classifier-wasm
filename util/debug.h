//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_DEBUG_H
#define FRUIT_CLASSIFIER_WASM_DEBUG_H

#ifdef DEBUG

#include <sstream>
#include <iostream>

static auto $FAST_IO = [] {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    return 0;
}();

#define PRINT(expr) { \
            std::ostringstream oss; \
            oss << expr; \
            std::cout << oss.str() << std::endl; \
        }

#define ASSERT(condition) \
        if (!(condition)) { \
            std::cerr << "Assertion failed in file " << __FILE__ << ", line " << __LINE__ << std::endl; \
            std::terminate(); \
        }

#define PRINT_ITER(prompt, iter) { \
    std::ostringstream oss; \
    oss << prompt << " ["; \
    for (auto i: (iter)) { oss << i << ","; } \
    oss << "]"; \
    std::cout << oss.str() << std::endl; \
}

#else
#define PRINT(expr)
#define ASSERT(condition)
#define PRINT_ITER(prompt, iter)
#endif

#endif //FRUIT_CLASSIFIER_WASM_DEBUG_H
