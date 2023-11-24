cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DDEBUG)

add_executable(fruit_classifier_wasm
        main.cpp
        util/debug.h
        lib/nn/act.h
        lib/nn/neuron.h
        lib/nn/layer.h
        lib/nn/neuron.cpp
        lib/nn/make.h
        lib/nn/layer.cpp
        lib/nn/nn.h)
