#pragma once

#include <vector>
#include <map>
#include <condition_variable>
#include "ISplitter.h"

class Splitter_impl : public ISplitter {
public:
    Splitter_impl(int _nMaxBuffers, int _nMaxClients);
    ~Splitter_impl() override = default;

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

    int push(const std::shared_ptr<std::vector<uint8_t>> &_data, int _ttl, int _nTimeOutMsec = 0);

private:
    std::shared_ptr<std::vector<uint8_t>> pop(int id, int index, int _nTimeOutMsec = 0);

    const int mMaxClients;

    static int sIdFrame;
    static int sIdClient;
    struct Item {
        Item() : _data(nullptr), _id(-1), _ttl(0) {}
        std::mutex _lock;
        std::condition_variable _cvWait;
        std::shared_ptr<std::vector<uint8_t>> _data;
        int _id;
        int _ttl;
    };
    struct Client {
        int _idFrame;
        int _indexFrame;
    };

    std::mutex mLockStorage;
    std::condition_variable mWaitStorageFrame;
    std::vector<Item> mStorageFrame;
    std::map<int, Item> mStorageClient;
    int mHeadStorageFrame, mTailStorageFrame;
    /* end guard mLockStorageFrame */
};