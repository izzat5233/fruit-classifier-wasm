cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

# Add nn directory to the build
add_subdirectory(nn)

# Add test directory to the build
add_subdirectory(test)

# Assuming GoogleTest is downloaded and placed in the 'lib/googletest' directory
add_subdirectory(lib/googletest)

# Enable testing with GoogleTest
enable_testing()
