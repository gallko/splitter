#include <ISplitter.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "../include/definitions.h"

class MultiThreadedTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        splitter = SplitterCreate(2, 5);
    }
    std::shared_ptr<ISplitter> splitter;

    std::mutex tactLock;
    std::condition_variable tackCv;
    int step = 0;

    void manager_behavior();
    void source_behavior();
    void client1_behavior();
    void client2_behavior();
};


TEST_F(MultiThreadedTest, FullTest) {


    std::thread manager([this](){ manager_behavior(); });
    std::thread source([this](){ source_behavior(); });
    std::thread client1([this](){ client1_behavior(); });
    std::thread client2([this](){ client2_behavior(); });

// step 1
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 2
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 3
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 4
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 5
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 6
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 7
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 8
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 9
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();
// step 10
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();

// exit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++step;
    tackCv.notify_all();

    source.join();
    client1.join();
    client2.join();
}


void MultiThreadedTest::manager_behavior() {
    while (step < 11) {
        std::unique_lock lock(tactLock);
        tackCv.wait(lock);
        lock.unlock();
        std::chrono::milliseconds(20);
        switch (step) {

        }
    }
}

void MultiThreadedTest::source_behavior() {
    while (step < 11) {
        std::unique_lock lock(tactLock);
        tackCv.wait(lock);
        lock.unlock();
        switch (step) {

        }
    }
}

void MultiThreadedTest::client1_behavior() {
    while (step < 11) {
        std::unique_lock lock(tactLock);
        tackCv.wait(lock);
        lock.unlock();
        switch (step) {

        }
    }
}

void MultiThreadedTest::client2_behavior() {
    while (step < 11) {
        std::unique_lock lock(tactLock);
        tackCv.wait(lock);
        lock.unlock();
        switch (step) {

        }
    }
}
