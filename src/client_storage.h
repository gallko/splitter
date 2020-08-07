#pragma once

#include <memory>
#include <map>
#include <shared_mutex>

class Frame;

class ClientStorage {
public:
    class Client;
    explicit ClientStorage(int _nMaxClients);

    int createClient(const std::shared_ptr<Frame> &_currentFrame);
    bool removeClient(const int &_nClientID);

    std::shared_ptr<Client> getClient(const int &_nClientID) const;

    void resetAll();

private:
    int mMaxClients;
    std::map<int, std::shared_ptr<Client>> mStorage;
    mutable std::shared_mutex mLockStorage;
};


class ClientStorage::Client {
public:
    explicit Client(const std::shared_ptr<Frame> &_currentFrame);

    std::shared_ptr<std::vector<uint8_t>> waitData(const int &_nTimeOutMsec);

    void reset() {};

private:
    std::shared_ptr<Frame> mCurrentFrame;
};

