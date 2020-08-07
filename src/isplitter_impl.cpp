#include <chrono>

#include "isplitter_impl.h"
#include "frame_buffer.h"
#include "client_storage.h"

std::shared_ptr<ISplitter> SplitterCreate(IN int _nMaxBuffers, IN int _nMaxClients) {
    return std::make_shared<Splitter_impl>(_nMaxBuffers, _nMaxClients);
}

Splitter_impl::Splitter_impl(int _nMaxBuffers, int _nMaxClients)
{

}

bool Splitter_impl::SplitterInfoGet(int *_pnMaxBuffers, int *_pnMaxClients) {
    return true;
}

int Splitter_impl::SplitterPut(const std::shared_ptr<std::vector<uint8_t>> &_pVecPut, int _nTimeOutMsec) {
    return 0;
}

int Splitter_impl::SplitterGet(int _nClientID, std::shared_ptr<std::vector<uint8_t>> &_pVecGet, int _nTimeOutMsec) {
    return 0;
}

int Splitter_impl::SplitterFlush() {
    return 0;
}

bool Splitter_impl::SplitterClientAdd(int *_pnClientID) {
    return false;
}

bool Splitter_impl::SplitterClientRemove(int _nClientID) {
    return false;
}

bool Splitter_impl::SplitterClientGetCount(int *_pnCount) {
    return false;
}

bool Splitter_impl::SplitterClientGetByIndex(int _nIndex, int *_pnClientID, int *_pnLatency) {
    return false;
}

void Splitter_impl::SplitterClose() {

}
