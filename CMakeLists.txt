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

# Add an executable for the main.cpp
add_executable(main_executable main.cpp)

# Include directories for nn library
target_include_directories(main_executable PUBLIC ${PROJECT_SOURCE_DIR}/nn)

# Link the executable to the necessary libraries
target_link_libraries(main_executable nn_lib)