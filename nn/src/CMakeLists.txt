# Collect all source files
file(GLOB_RECURSE NN_SOURCES "*.cpp")

# Create a library from the source files
add_library(nn_lib STATIC ${NN_SOURCES})
