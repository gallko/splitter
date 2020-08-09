#include <iostream>
#include "ISplitter.h"
#include "src/isplitter_impl.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    int idClient = 0;
    std::shared_ptr<std::vector<uint8_t>> _pVecGet;

    auto sp = Splitter_impl(5,10);

    sp.SplitterClientAdd(&idClient);
//int _nClientID, std::shared_ptr<std::vector<uint8_t>> &_pVecGet, int _nTimeOutMsec
    std::cout << "SplitterPut\n";
    sp.SplitterPut(_pVecGet, 1000);
    sp.SplitterPut(_pVecGet, 1000);
    sp.SplitterPut(_pVecGet, 1000);
    std::cout << "SplitterGet\n";
    sp.SplitterGet(idClient, _pVecGet, 1000);
    std::cout << "1\n";
    sp.SplitterGet(idClient, _pVecGet, 1000);
    std::cout << "2\n";
    sp.SplitterGet(idClient, _pVecGet, 1000);
    std::cout << "3\n";
    sp.SplitterGet(idClient, _pVecGet, 1000);
    std::cout << "end\n";

    return 0;
}
