cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DDEBUG)

add_executable(fruit_classifier_wasm
        main.cpp
        util/debug.h
        nn/layer/hidden_layer.h
        nn/neuron/neuron.cpp
        nn/make/make.h
        nn/layer/layer.cpp
        nn/nn.h
        nn/layer/output_layer.h
        nn/neuron/neuron.h
        nn/layer/layer.h
        nn/act/act.cpp
        nn/make/make.cpp
        nn/network/network.h
        nn/network/network.cpp
)
