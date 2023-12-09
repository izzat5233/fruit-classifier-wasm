# Collect all test source files
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

# Define the test executable
add_executable(test_nn ${TEST_SOURCES}
        layer_test.cpp
        act_test.cpp
        loss_test.cpp
        network_test.cpp
        globals.h
)

# Link the GoogleTest libraries and the neural network library
target_link_libraries(test_nn gtest gtest_main nn_lib)

# Include directories for GoogleTest and nn library
target_include_directories(test_nn PUBLIC
        ${PROJECT_SOURCE_DIR}/nn
        ${PROJECT_SOURCE_DIR}/lib/googletest/include
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(test_nn)
