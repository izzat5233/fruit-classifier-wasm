cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    # Wasm files directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web/static/wasm)
endif ()

# Add test directory to the build only for Debug configuration
if (CMAKE_BUILD_TYPE MATCHES Debug)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
    )
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(test)
    enable_testing()
endif ()

# Add nn directory to the build
add_subdirectory(nn)

# Add an executable for the main.cpp
add_executable(main_executable main.cpp)

# Include directories for nn library
target_include_directories(main_executable PUBLIC ${PROJECT_SOURCE_DIR}/nn)

# Link the executable to the necessary libraries
target_link_libraries(main_executable nn_lib)

# Set different output names for different configurations
set_target_properties(main_executable PROPERTIES OUTPUT_NAME "wasm" DEBUG_POSTFIX "_debug")

# Set compiler options and definitions based on the build type
if (CMAKE_BUILD_TYPE MATCHES Debug)
    # Debugging flags
    target_compile_definitions(main_executable PRIVATE DEBUG=1)
    target_link_options(main_executable PRIVATE "SHELL:-s ASSERTIONS=1")
else ()
    # Release flags
    target_link_options(main_executable PRIVATE "SHELL:-s NO_EXIT_RUNTIME=1")
    target_link_options(main_executable PRIVATE "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall']")
endif ()
