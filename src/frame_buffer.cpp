#include "frame_buffer.h"

#include <utility>

Frame::Frame(int _nMaxClients, std::condition_variable &_mCvWaitData, std::mutex &_mLockWaitData, std::weak_ptr<std::vector<uint8_t>> _data)
    : mData(std::move(_data))
    , mCvWaitData(_mCvWaitData)
    , mLockWaitData(_mLockWaitData)
    , mReasonWeakUpWaitData(ReasonWeakUp::FAKE)
    , mCvWaitPickUp()
    , mLockWaitPickUp()
    , mReasonWeakUpPickUp(ReasonWeakUp::FAKE)
    , mCountUse(_nMaxClients)
{

}
