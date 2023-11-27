cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DDEBUG)

add_executable(fruit_classifier_wasm
        main.cpp
        util/debug.h
        nn/hidden_layer.h
        src/neuron.cpp
        src/layer.cpp
        nn/nn.h
        nn/output_layer.h
        nn/neuron.h
        nn/layer.h
        src/act.cpp
        src/make.cpp
        nn/network.h
        src/network.cpp
)
