cmake_minimum_required(VERSION 3.26)
project(fruit_classifier_wasm)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DDEBUG)

add_executable(fruit_classifier_wasm
        main.cpp
        util/debug.h
        nn/act.h
        nn/neuron/neuron.h
        nn/layer/layer.h
        nn/neuron/neuron.cpp
        nn/make.h
        nn/layer/layer.cpp
        nn/nn.h)
