#include <iostream>
#include "ISplitter.h"
#include "src/frame_buffer.h"
#include "src/isplitter_impl.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto sp = Splitter_impl(0,10);


    sp.push(nullptr, 10, 50);
    sp.push(nullptr, 10, 50);

    sp.push(nullptr, 10, 50);
    sp.push(nullptr, 10, 50);
    sp.push(nullptr, 10, 50);


    return 0;
}
