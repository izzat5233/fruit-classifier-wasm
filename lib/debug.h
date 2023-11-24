//
// Created by Izzat on 11/24/2023.
//

#ifndef FRUIT_CLASSIFIER_WASM_DEBUG_H
#define FRUIT_CLASSIFIER_WASM_DEBUG_H

#ifdef DEBUG
#include <iostream>
#define PRINT(str) std::cout << (str) << std::endl;
#define ASSERT(condition) \
        if (!(condition)) { \
            std::cerr << "Assertion failed in file " << __FILE__ << ", line " << __LINE__ << std::endl; \
            std::terminate(); \
        }
#else
#define PRINT(str)
#define ASSERT(condition)
#endif


#endif //FRUIT_CLASSIFIER_WASM_DEBUG_H
