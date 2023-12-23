cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    # Wasm files directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web/static/wasm)
endif ()

# Add nn directory to the build
add_subdirectory(nn)

# Add test directory to the build only for Debug configuration
if (CMAKE_BUILD_TYPE MATCHES Debug)
    # Fetch googletest
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    # Add test directory to build
    add_subdirectory(test)
else ()
    # Add an executable for the main.cpp
    add_executable(main_executable interface.cpp)

    # Include directories for nn library
    target_include_directories(main_executable PUBLIC ${PROJECT_SOURCE_DIR}/nn)

    # Link the executable to the necessary libraries
    target_link_libraries(main_executable nn_lib)

    # Set output name
    set_target_properties(main_executable PROPERTIES OUTPUT_NAME "main")

    # Release flags
    target_link_options(main_executable PRIVATE "SHELL:--bind")
    target_link_options(main_executable PRIVATE "SHELL:-s NO_EXIT_RUNTIME=1")
    target_link_options(main_executable PRIVATE "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']")
    target_link_options(main_executable PRIVATE "SHELL:-s EXPORTED_FUNCTIONS=[_main,_malloc,_free]")
    target_link_options(main_executable PRIVATE "SHELL:-s ALLOW_MEMORY_GROWTH=1")
endif ()
