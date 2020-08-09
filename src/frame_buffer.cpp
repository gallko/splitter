#include "frame_buffer.h"

#include <utility>
#include <iostream>

int FrameBuffer::sIdFrame = 0;

FrameBuffer::FrameBuffer(int _maxCount)
    : mCountFrame(0)
    , mCountClient(0)
    , mMaxBuffers(_maxCount)
{
    mStorage.resize(_maxCount + 1);
    mHead = mTail = mStorage.begin();
}

//void FrameBuffer::push(const UserData &_data, int _nTimeOutMsec) {
//    if ((mTail - mHead) == 1 || (mHead - mTail) == mMaxBuffers) {
//        std::cout << "Buffer is full\n";
//        return;
//    }
//
//    mHead->_data = _data;
//    mHead->_ttl = 0;
//    mHead->_id = sIdFrame;
//
//    if (++sIdFrame < 0) {
//        sIdFrame = 0;
//    }
//    if (++mHead == mStorage.end()) {
//        mHead = mStorage.begin();
//    }
//}
//
//FrameBuffer::UserData FrameBuffer::pop(int _nTimeOutMsec) {
//    if (mTail == mHead) {
//        std::cout << "Buffer is empty\n";
//        return nullptr;
//    }
//
//    mTail->_id = -1;
//    mTail->_ttl = 0;
//    if (++mTail == mStorage.end()) {
//        mTail = mStorage.begin();
//    }
//
//    return mTail->_data;
//}

int FrameBuffer::countFrame() const {
    return mTail <= mHead ? mHead - mTail : (mStorage.end() - mTail) + (mHead - mStorage.begin());
}

int FrameBuffer::maxCountFrame() const {
    return mMaxBuffers;
}
