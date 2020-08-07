#pragma once

#include <vector>
#include <memory>
#include <condition_variable>

class Frame {
public:
    enum class ReasonWeakUp;

    Frame(int _nMaxClients, std::condition_variable &_mCvWaitData, std::mutex &mLockWaitData, std::weak_ptr<std::vector<uint8_t>> _data);

    std::weak_ptr<std::vector<uint8_t>> getData() const;

    std::shared_ptr<Frame> next();

private:
    std::weak_ptr<std::vector<uint8_t>> mData;

// waiting data from the source (users are waiting for data when it arrives)
    std::condition_variable &mCvWaitData;
    std::mutex &mLockWaitData;
    ReasonWeakUp mReasonWeakUpWaitData;

// waiting users for pick up the data (the storage is full)
    std::condition_variable mCvWaitPickUp;
    std::mutex mLockWaitPickUp;
    ReasonWeakUp mReasonWeakUpPickUp;
    int mCountUse;
};

enum class Frame::ReasonWeakUp {
    FAKE, OK, RESET
};

inline std::weak_ptr<std::vector<uint8_t>> Frame::getData() const {
    return mData;
}
