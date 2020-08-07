#pragma once

#include <list>
#include <condition_variable>
#include "ISplitter.h"

class Splitter_impl : public ISplitter {
public:
    Splitter_impl(int _nMaxBuffers, int _nMaxClients);

/* class ISplitter */
    bool SplitterInfoGet(OUT int *_pnMaxBuffers, OUT int *_pnMaxClients) override;
    int SplitterPut(IN const std::shared_ptr <std::vector<uint8_t>> &_pVecPut, IN int _nTimeOutMsec) override;
    int SplitterFlush() override;
    bool SplitterClientAdd(OUT int *_pnClientID) override;
    bool SplitterClientRemove(IN int _nClientID) override;
    bool SplitterClientGetCount(OUT int *_pnCount) override;
    bool SplitterClientGetByIndex(IN int _nIndex, OUT int *_pnClientID, OUT int *_pnLatency) override;
    int SplitterGet(IN int _nClientID, OUT std::shared_ptr<std::vector<uint8_t>> &_pVecGet, IN int _nTimeOutMsec) override;
    void SplitterClose() override;

private:

    std::list< std::shared_ptr<std::vector<uint8_t>> > mStorage;
    std::mutex  mLockStorage;
    std::condition_variable mCvStorage;
};