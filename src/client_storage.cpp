#include <algorithm>

#include "client_storage.h"
#include "frame_buffer.h"

ClientStorage::ClientStorage(int _nMaxClients)
    : mMaxClients(_nMaxClients)
{

}

int ClientStorage::createClient(const std::shared_ptr<Frame> &_currentFrame) {
    static int ID = 0;
    int result = -1;

    std::unique_lock<std::shared_mutex> lk(mLockStorage);
    if (mStorage.size() < mMaxClients) {
        mStorage[++ID] = std::make_shared<Client>(_currentFrame);
        result = ID;
    }

    return result;
}

bool ClientStorage::removeClient(const int &_nClientID) {
    bool result = false;

    std::unique_lock<std::shared_mutex> lk(mLockStorage);
    if (auto it = mStorage.find(_nClientID); it != mStorage.end()) {
        mStorage.erase(it);
        result = true;
    }

    return result;
}

std::shared_ptr<ClientStorage::Client> ClientStorage::getClient(const int &_nClientID) const {
    std::shared_lock<std::shared_mutex> lk(mLockStorage);
    auto it = mStorage.find(_nClientID);

    return it != mStorage.end() ? it->second : nullptr;
}

void ClientStorage::resetAll() {
    std::shared_lock<std::shared_mutex> lk(mLockStorage);
    std::for_each(mStorage.begin(), mStorage.end(), [](auto &it) {
        it.second->reset();
    });
}

ClientStorage::Client::Client(const std::shared_ptr<Frame> &_currentFrame)
    : mCurrentFrame(_currentFrame)
{

}

std::shared_ptr<std::vector<uint8_t>> ClientStorage::Client::waitData(const int &_nTimeOutMsec) {
//    auto data = mCurrentFrame->getData().lock();

//    if (!data) {
//        // wait data
//
//        data = mCurrentFrame->getData().lock();
//    }

    return nullptr;
}

