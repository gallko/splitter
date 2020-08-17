#include <chrono>
#include <iostream>
#include "isplitter_impl.h"

int Splitter_impl::sIdClient = 0;

std::shared_ptr<ISplitter> SplitterCreate(IN int _nMaxBuffers, IN int _nMaxClients) {
    return std::make_shared<Splitter_impl>(_nMaxBuffers, _nMaxClients);
}

bool Splitter_impl::SplitterInfoGet(int *_pnMaxBuffers, int *_pnMaxClients) {
    if (mIsOpen.load()) {
        *_pnMaxBuffers = mMaxBuffers;
        *_pnMaxClients = mMaxClients;
        return true;
    }
    return false;
}

int Splitter_impl::SplitterPut(const std::shared_ptr<std::vector<uint8_t>> &_pVecPut, int _nTimeOutMsec) {
    int result = CLOSED;
    if (mIsOpen.load()) {
        std::shared_lock<std::shared_mutex> lockClient(mLockClient);
        if (!mStorageClient.empty()) {
            lockClient.unlock();
            result = push(_pVecPut, mStorageClient.size(), _nTimeOutMsec);
        } else {
            result = SUCCESS;
        }
    }
    return result;
}

int Splitter_impl::SplitterGet(int _nClientID, std::shared_ptr<std::vector<uint8_t>> &_pVecGet, int _nTimeOutMsec) {
    int result = CLOSED;
    if (mIsOpen.load()) {
        std::shared_lock<std::shared_mutex> lockClient(mLockClient);
        auto it = mStorageClient.find(_nClientID);
        if (it != mStorageClient.end()) {
            result = extract_data(it->second, _pVecGet, _nTimeOutMsec);
        } else {
            result = CLIENT_NOT_FOUND;
        }
    }
    return result;
}

int Splitter_impl::SplitterFlush() {
    int result = CLOSED;
    if (mIsOpen.load()) {
        std::unique_lock<std::shared_mutex> lockClient(mLockClient);
        std::unique_lock<std::mutex> lockStorage(mLockStorage);
        for (auto it: mStorageClient) {
            it.second = mHead;
        }
        while (mTail != mHead) {
            mTail = mTail->mNext;
        }
        mSize = 0;
        mWakeUpReasonHead = ReasonWakeUp::reset;
        mWakeUpReasonTail = ReasonWakeUp::reset;
        mWaitHead.notify_all();
        mWaitTail.notify_all();
        result = SUCCESS;
    }
    return result;
}

bool Splitter_impl::SplitterClientAdd(int *_pnClientID) {
    bool result = false;
    if (mIsOpen.load()) {
        std::unique_lock<std::shared_mutex> lockClient(mLockClient);
        if (mStorageClient.size() != mMaxClients) {
            std::unique_lock<std::mutex> lockStorage(mLockStorage);
            *_pnClientID = ++sIdClient;
            mStorageClient[*_pnClientID] = mHead;
            result = true;
        }
    }
    return result;
}

bool Splitter_impl::SplitterClientRemove(int _nClientID) {
    bool result = false;
    if (mIsOpen.load()) {
        std::unique_lock<std::shared_mutex> lockClient(mLockClient);
        auto it = mStorageClient.find(_nClientID);
        if (it != mStorageClient.end()) {
            mLockStorage.lock();
            auto removedItem = it->second.lock();
            auto endRemoved = mHead;
            it->second.reset();
            mStorageClient.erase(it);

            mWakeUpReasonHead = ReasonWakeUp::removed_client;
            mWakeUpReasonTail = ReasonWakeUp::removed_client;
            mWaitHead.notify_all();
            mWaitTail.notify_all();
            mLockStorage.unlock();
            lockClient.unlock();

            while (removedItem != endRemoved) {
                if (removedItem->mTtl.fetch_sub(1) == 1) {
                    std::unique_lock<std::mutex> lockStorage(mLockStorage);
                    mTail = mTail->mNext;
                }
                removedItem = removedItem->mNext;
            }
            result = true;
        }
    }
    return result;
}

bool Splitter_impl::SplitterClientGetCount(int *_pnCount) {
    if (mIsOpen.load()) {
        std::shared_lock<std::shared_mutex> lockClient(mLockClient);
        *_pnCount = mStorageClient.size();
        return true;
    }
    return false;
}

bool Splitter_impl::SplitterClientGetByIndex(int _nIndex, int *_pnClientID, int *_pnLatency) {
    bool result = false;
    if (mIsOpen.load()) {
        std::shared_lock<std::shared_mutex> lockClient(mLockClient);
        if (_nIndex < mStorageClient.size()) {
            auto client = std::next(mStorageClient.begin(), _nIndex);
            *_pnClientID = client->first;

            mLockStorage.lock();
            auto head = mHead;
            auto tail = client->second.lock();
            if (!tail) {
                client->second = tail = mTail;
            }
            mLockStorage.unlock();

            *_pnLatency = 0;
            while (tail != head) {
                tail = tail->mNext;
                ++*_pnLatency;
            }
            result = true;
        }
    }
    return result;
}

void Splitter_impl::SplitterClose() {
    mIsOpen.store(false );
    {
        std::unique_lock<std::mutex> lockStorage(mLockStorage);
        mWakeUpReasonHead = ReasonWakeUp::closed;
        mWakeUpReasonTail = ReasonWakeUp::closed;
        mWaitHead.notify_all();
        mWaitTail.notify_all();
    }

    std::unique_lock lockClient(mLockClient);
    std::unique_lock lockStorage(mLockStorage);
    while (mTail != mHead) {
        mTail = mTail->mNext;
    }
    mSize = 0;
    mStorageClient.clear();
}

/**
 *
 * */
Splitter_impl::Splitter_impl(int _nMaxBuffers, int _nMaxClients)
        : mMaxClients(_nMaxClients)
        , mMaxBuffers(_nMaxBuffers)
        , mIsOpen(true)
        , mWakeUpReasonHead(ReasonWakeUp::fake)
        , mWakeUpReasonTail(ReasonWakeUp::fake)
        , mLockStorage()
        , mHead(nullptr)
        , mTail(nullptr)
        , mSize(0)
        , mStorageClient()
{
        mHead = mTail = std::make_shared<Item>();
}

Splitter_impl::~Splitter_impl() {
    Splitter_impl::SplitterClose();
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
        mWakeUpReasonTail = ReasonWakeUp::fake;
        // waiting that the last element will read
        auto flag = mWaitTail.wait_for(lockStorage, std::chrono::milliseconds(_nTimeOutMsec), [this, &code](){
            switch (mWakeUpReasonTail) {
                case ReasonWakeUp::removed_client:
                case ReasonWakeUp::added_item:
                case ReasonWakeUp::fake: return false;
                case ReasonWakeUp::removed_item: return true;
                case ReasonWakeUp::reset: code = FLUSH; return true;
                case ReasonWakeUp::closed: code = CLOSED; return true;
            }
            return false;
        });

        if (!flag) { // the last element wasn't read, delete it
            mTail = mTail->mNext;
            --mSize;
            code = CLIENT_MISSED_DATA;
        }
    }

    if (code == SUCCESS || code == CLIENT_MISSED_DATA) {
        auto ptr = std::move(mHead);
        ptr->mNext = std::make_shared<Item>();
        ptr->mData = _data;
        ptr->mTtl.store(_ttl, std::memory_order_release);
        mHead = ptr->mNext;

        ++mSize;

        mWakeUpReasonHead = ReasonWakeUp::added_item;
        lockStorage.unlock();
        mWaitHead.notify_all();
    }
    return code;
}

int Splitter_impl::extract_data(std::weak_ptr<Item> &_item, std::shared_ptr<std::vector<uint8_t>> &_data, int _nTimeOutMsec) {
    ResultCode code = SUCCESS;
    auto item = _item.lock();

    {
        std::unique_lock<std::mutex> lockStorage(mLockStorage);
        if (!item) {
            item = mTail;
        }
        if (item == mHead) { // is storage empty?
            mWakeUpReasonHead = ReasonWakeUp::fake;
            // waiting to come data
            mLockClient.unlock();
            auto flag = mWaitHead.wait_for(lockStorage, std::chrono::milliseconds(_nTimeOutMsec), [this, &code, &_item](){
                switch (mWakeUpReasonHead) {
                    case ReasonWakeUp::removed_item:
                    case ReasonWakeUp::fake: return false;
                    case ReasonWakeUp::added_item: return true;
                    case ReasonWakeUp::reset: code = FLUSH; return true;
                    case ReasonWakeUp::closed: code = CLOSED; return true;
                    case ReasonWakeUp::removed_client:
                        if (_item.expired()) {
                            code = CLIENT_REMOVED;
                            return true;
                        } else {
                            return false;
                        }
                }
                return false;
            });
            mLockClient.lock_shared();
            if (!flag) { // an element wasn't add
                code = TIME_OUT;
            }
        }
    }
    if (code == SUCCESS) {
        if (item->mTtl.load(std::memory_order_acquire) == 1) { // clean up after yourself
            {
                std::unique_lock<std::mutex> lockStorage(mLockStorage);
                mTail = mTail->mNext;
                --mSize;
                mWakeUpReasonTail = ReasonWakeUp::removed_item;
                mWaitTail.notify_all();
            }
        } else {
            --item->mTtl;
        }
        _data = item->mData;
        _item = item->mNext;
    }
    return code;
}
