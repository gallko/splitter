#pragma once

#include <atomic>
#include <vector>
#include <map>
#include <list>
#include <condition_variable>
#include <shared_mutex>
#include <ISplitter.h>

class Splitter_impl : public ISplitter {
    enum class ReasonWakeUp;
    struct Item;

public:
    Splitter_impl(int _nMaxBuffers, int _nMaxClients);
    ~Splitter_impl() override;

    Splitter_impl(const Splitter_impl&) = delete;
    Splitter_impl(Splitter_impl&&) = delete;
    Splitter_impl &operator=(const Splitter_impl&) = delete;
    Splitter_impl &operator=(Splitter_impl&&) = delete;

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
    int push(const std::shared_ptr<std::vector<uint8_t>> &_data, int _ttl, int _nTimeOutMsec = 0);
    int extract_data(std::weak_ptr<Item> &_item, std::shared_ptr<std::vector<uint8_t>> &_data, int _nTimeOutMsec);

private:
    const int mMaxClients;
    const int mMaxBuffers;

    static int sIdClient;

    std::atomic<bool> mIsOpen;

    ReasonWakeUp mWakeUpReasonHead, mWakeUpReasonTail;
    std::mutex mLockStorage;
    std::condition_variable mWaitHead, mWaitTail;
    std::shared_ptr<Item> mHead, mTail;
    int mSize;

    std::shared_mutex mLockClient;
    std::map<int, std::weak_ptr<Item>> mStorageClient;
};

enum class Splitter_impl::ReasonWakeUp {
    added_item, removed_item, reset, closed, removed_client, fake
};

struct Splitter_impl::Item {
    Item() : mData(nullptr), mNext(nullptr), mTtl(0) {}
//    std::shared_mutex mLock;
    std::shared_ptr<std::vector<uint8_t>> mData;
    std::shared_ptr<Item> mNext;
    std::atomic<int> mTtl;
};
