#include <emscripten/emscripten.h>
#include <network.h>

extern "C" {
EMSCRIPTEN_KEEPALIVE int add(int a, int b) {
    return a + b;
}
}

int main() {
    return 0;
}