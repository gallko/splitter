#include <chrono>
#include <iostream>
#include "isplitter_impl.h"

int Splitter_impl::sIdClient = 0;

std::shared_ptr<ISplitter> SplitterCreate(IN int _nMaxBuffers, IN int _nMaxClients) {
    return std::make_shared<Splitter_impl>(_nMaxBuffers, _nMaxClients);
}

bool Splitter_impl::SplitterInfoGet(int *_pnMaxBuffers, int *_pnMaxClients) {
    *_pnMaxBuffers = mMaxBuffers;
    *_pnMaxClients = mMaxClients;
    return true;
}

int Splitter_impl::SplitterPut(const std::shared_ptr<std::vector<uint8_t>> &_pVecPut, int _nTimeOutMsec) {
    int result = CLOSED;
    if (mIsOpen.load(std::memory_order_acquire)) {
        result = push(_pVecPut, mStorageClient.size(), _nTimeOutMsec);
    }
    return result;
}

int Splitter_impl::SplitterGet(int _nClientID, std::shared_ptr<std::vector<uint8_t>> &_pVecGet, int _nTimeOutMsec) {
    int result;
    if (mIsOpen.load(std::memory_order_acquire)) {
        auto it = mStorageClient.find(_nClientID);
        if (it != mStorageClient.end()) {
            result = extract_data(it->second, _pVecGet, _nTimeOutMsec);
        } else {
            result = CLIENT_NOT_FOUND;
        }
    } else {
        result = CLOSED;
    }
    return result;
}

int Splitter_impl::SplitterFlush() {
    int result = CLOSED;
    if (mIsOpen.load(std::memory_order_acquire)) {
        flushBuffer();
        result = SUCCESS;
    }
    return result;
}

bool Splitter_impl::SplitterClientAdd(int *_pnClientID) {
    bool result = false;
    if (mIsOpen.load(std::memory_order_acquire)) {
        std::unique_lock<std::shared_mutex> lockClient(mLockClient);
        std::unique_lock<std::mutex> lockStorage(mLockStorage);
        *_pnClientID = ++sIdClient;
        mStorageClient[*_pnClientID] = mHead;
        result = SUCCESS;
    }
    return result;
}

bool Splitter_impl::SplitterClientRemove(int _nClientID) {
    bool result = false;
    if (mIsOpen.load(std::memory_order_acquire)) {
        std::unique_lock<std::shared_mutex> lock(mLockClient);
        auto it = mStorageClient.find(_nClientID);
        if (it != mStorageClient.end()) {
            std::unique_lock<std::mutex> lockStorage(mLockStorage);

            auto removedItem = it->second;
            mLockStorage.lock();
            mStorageClient.erase(it);
            auto endRemoved = mHead;
            mLockStorage.unlock();

            // decrease all ttl [removedItem, endRemoved)
        }
    }
    return result;
}

bool Splitter_impl::SplitterClientGetCount(int *_pnCount) {
    std::unique_lock<std::mutex> lockStorage(mLockStorage);
    *_pnCount = mSize;
    return true;
}

bool Splitter_impl::SplitterClientGetByIndex(int _nIndex, int *_pnClientID, int *_pnLatency) {
    return false;
}

void Splitter_impl::SplitterClose() {

}

/**
 *
 * */
Splitter_impl::Splitter_impl(int _nMaxBuffers, int _nMaxClients)
        : mIsOpen(true)
        , mMaxClients(_nMaxClients)
        , mMaxBuffers(_nMaxBuffers)
        , mLockStorage()
        , mWakeUpHead()
        , mWakeUpTail()
        , mHead(nullptr)
        , mTail(nullptr)
        , mSize(0)
        , mStorageClient()
{
        mHead = mTail = std::make_shared<Item>();
}

int Splitter_impl::push(const std::shared_ptr<std::vector<uint8_t>> &_data, int _ttl, int _nTimeOutMsec) {
    ResultCode code = SUCCESS;
    std::unique_lock<std::mutex> lockStorage(mLockStorage);

//    management storage
    while (mTail != mHead && !mTail->mTtl.load(std::memory_order_acquire)) {
        --mSize;
        mTail = mTail->mNext;
    }

    if (mSize == mMaxBuffers) {
        std::cout << "Storage is full. Wait..." << std::endl;
        mWakeUpTail = ReasonWakeUp::fake;
        // waiting that the last element will read
        auto flag = mWaitTail.wait_for(lockStorage, std::chrono::milliseconds(_nTimeOutMsec), [this](){
            return mWakeUpTail == ReasonWakeUp::removed_item;
        });
        if (!flag) { // the last element wasn't read, delete it
            mTail = mTail->mNext;
            --mSize;
            code = CLIENT_MISSED_DATA;
        }
    }

    mHead->mNext = std::make_shared<Item>();
    mHead->mData = _data;
    mHead->mTtl.store(_ttl, std::memory_order_release);

    mHead = mHead->mNext;
    ++mSize;

    mWakeUpHead = ReasonWakeUp::added_item;
    lockStorage.unlock();
    mWaitHead.notify_all();
    return code;
}

int Splitter_impl::extract_data(std::weak_ptr<Item> &_item, std::shared_ptr<std::vector<uint8_t>> &_data, int _nTimeOutMsec) {
    auto item = _item.lock();
    if (!item) {
        std::unique_lock<std::mutex> lock(mLockStorage);
        item = mTail;
    }

    /* ttl is expired, is it bug? */
//    if (!item->mTtl.load(std::memory_order_acquire)) {
//        assert(0);
//    }

    { // storage empty, waiting to come data
        std::unique_lock<std::mutex> lockStorage(mLockStorage);
        if (item == mHead) {
            mWakeUpHead = ReasonWakeUp::fake;
            // waiting that the last element will read
            auto flag = mWaitHead.wait_for(lockStorage, std::chrono::milliseconds(_nTimeOutMsec), [this](){
                return mWakeUpHead == ReasonWakeUp::added_item;
            });

            if (!flag) { // an element wasn't add
                return TIME_OUT;
            }
        }
    }

    if (item->mTtl.load(std::memory_order_acquire) == 1) { // clean up after yourself
        {
            std::unique_lock<std::mutex> lockStorage(mLockStorage);
            mTail = mTail->mNext;
            --mSize;
            mWakeUpTail = ReasonWakeUp::removed_item;
        }
        mWaitTail.notify_all();
    }
    _data = item->mData;
    _item = item->mNext;
    return SUCCESS;
}
