#include <chrono>
#include <iostream>

#include "isplitter_impl.h"

namespace {
    inline int increaseIndexFrame(int currentValue, int maxValue) {
        if (++currentValue == maxValue) {
            currentValue = 0;
        }
        return currentValue;
    }
}

int Splitter_impl::sIdFrame = 0;
int Splitter_impl::sIdClient = 0;

std::shared_ptr<ISplitter> SplitterCreate(IN int _nMaxBuffers, IN int _nMaxClients) {
    return std::make_shared<Splitter_impl>(_nMaxBuffers, _nMaxClients);
}

Splitter_impl::Splitter_impl(int _nMaxBuffers, int _nMaxClients)
    : mMaxClients(_nMaxClients)
    , mLockStorage()
    , mWaitStorageFrame()
    , mStorageFrame(_nMaxBuffers + 1)
    , mStorageClient()
    , mHeadStorageFrame(0)
    , mTailStorageFrame(0)
{
    if (_nMaxBuffers != 0) {
        ++mHeadStorageFrame;
    }
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

int Splitter_impl::push(const std::shared_ptr<std::vector<uint8_t>> &_data, int _ttl, int _nTimeOutMsec) {
    std::unique_lock<std::mutex> lock(mLockStorage);
    if (mTailStorageFrame == mHeadStorageFrame) {
        std::cout << "Buffer is full, wait...\n";
        return 0;
    }

    mStorageFrame[mHeadStorageFrame]._ttl = _ttl;
    mStorageFrame[mHeadStorageFrame]._data = _data;
    mStorageFrame[mHeadStorageFrame]._id = sIdFrame;

    mHeadStorageFrame = increaseIndexFrame(mHeadStorageFrame, mStorageFrame.size());

    if (++sIdFrame < 0) {
        sIdFrame = 0;
    }
    lock.unlock();
    mWaitStorageFrame.notify_all();

    return 0;
}
