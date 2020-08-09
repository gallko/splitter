#pragma once

#include <vector>
#include <memory>
#include <condition_variable>

class FrameBuffer {
public:
    class IClient;
    using UserData = std::shared_ptr<std::vector<uint8_t>>;

    explicit FrameBuffer(int _maxCount);

    int push(const UserData &_data, int _ttl, int _nTimeOutMsec = 0);
    UserData pop(int id, int index, int _nTimeOutMsec = 0);

    int countFrame() const;
    int maxCountFrame() const;

private:
    static int sIdFrame;
    struct Item {
        Item() : _data(nullptr), _id(-1), _ttl(0) {}
        UserData _data;
        int _id;
        int _ttl;
    };
    std::vector<Item> mStorage;
    typename std::vector<Item>::iterator mHead, mTail;
    int mCountFrame;
    int mCountClient;
    const int mMaxBuffers;
};

/*
class FrameBuffer<T>::Frame {
    friend FrameBuffer;
public:
    enum class ReasonWeakUp;
    Frame(FrameBuffer &_frameBuffer):mFrameBuffer(_frameBuffer) {};

    std::weak_ptr<std::vector<uint8_t>> getData() const;
    std::shared_ptr<Frame> next();

private:
    FrameBuffer &mFrameBuffer;
    std::shared_ptr<Frame> mNext;
    std::shared_ptr<std::vector<uint8_t>> mData;

    int mNumbOfWaiting;

// waiting data from the source (users are waiting for data when it arrives)
//    std::condition_variable &mCvWaitData;
//    std::mutex &mLockWaitData;
//    ReasonWeakUp mReasonWeakUpWaitData;

// waiting users for pick up the data (the storage is full)
    std::condition_variable mCvWaitPickUp;
    std::mutex mLockWaitPickUp;
//    ReasonWeakUp mReasonWeakUpPickUp;
//    int mCountUse;
};

enum class FrameBuffer::Frame::ReasonWeakUp {
    FAKE, OK, RESET
};

inline std::weak_ptr<std::vector<uint8_t>> FrameBuffer::Frame::getData() const {
    return mData;
}
*/