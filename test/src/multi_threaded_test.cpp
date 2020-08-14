#include <ISplitter.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
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
    std::atomic<int> step;

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

    for (step = 0; step.load() <= 51; ++step) {
        tackCv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }



    step = -1;
    tackCv.notify_all();

    manager.join();
    source.join();
    client1.join();
    client2.join();
}

void MultiThreadedTest::manager_behavior() {
    int result_function, latency = -1;
    while (step != -1) {
        {
            std::unique_lock lock(tactLock);
            tackCv.wait(lock);
        }
        std::vector<uint8_t> data{static_cast<uint8_t>(step)};
        switch (step) {
            case 3:
                EXPECT_TRUE(splitter->SplitterClientGetCount(&result_function)) << "step: " << step;
                EXPECT_EQ(2, result_function) << "step: " << step;;
            case 15:
            case 43:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency);
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency) << "step: " << step;
                break;
            case 6:
            case 11:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(1, latency) << "step: " << step;;
                break;
            case 9:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(1, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(2, latency) << "step: " << step;;
                break;
            case 19:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(1, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(1, latency) << "step: " << step;;
                break;
            case 27:
            case 31:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(2, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency) << "step: " << step;;
                break;
            case 35:
            case 39:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(2, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(1, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(1, latency) << "step: " << step;;
                break;
            case 41:
                result_function = splitter->SplitterFlush();
                EXPECT_EQ(SUCCESS, result_function);
                break;
            case 47:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency) << "step: " << step;;
                EXPECT_TRUE(splitter->SplitterClientRemove(result_function));
                break;
            case 48:
                EXPECT_TRUE(splitter->SplitterClientGetByIndex(0, &result_function, &latency)) << "step: " << step;
                EXPECT_EQ(0, latency) << "step: " << step;;
                break;
            case 51:
                splitter->SplitterClose();
                return;
            default:
                break;
        }
    }
}

void MultiThreadedTest::source_behavior() {
    int time, result_function;
    while (step != -1) {
        {
            std::unique_lock lock(tactLock);
            tackCv.wait(lock);
        }
        std::vector<uint8_t> data{static_cast<uint8_t>(step)};
        switch (step) {
            case 0:
            case 4:
            case 8:
            case 12:
            case 16:
            case 20:
            case 25:
            case 37:
            case 44:
            case 49:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                break;
            case 42:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                break;
            case 28:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 50);
                EXPECT_EQ(CLIENT_MISSED_DATA, result_function) << "step: " << step;
                ASSERT_LE(time, 51) << "step: " << step;
                break;
            case 32:
            case 33:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 20);
                EXPECT_EQ(CLIENT_MISSED_DATA, result_function) << "step: " << step;
                ASSERT_LE(time, 21) << "step: " << step;
                break;
            case 40:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 50);
                EXPECT_EQ(FLUSH, result_function) << "step: " << step;
                ASSERT_LE(time, 50) << "step: " << step;
                break;
            case 52:
                MEASURE_CALL_R(time, result_function, splitter->SplitterPut, std::make_shared<std::vector<uint8_t>>(data), 20);
                EXPECT_EQ(CLOSED, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                break;
            default:
                break;
        }
    }
}

void MultiThreadedTest::client1_behavior() {
    int time, result_function, clientID;
    std::shared_ptr <std::vector<uint8_t>> outData;
    while (step != -1) {
        {
            std::unique_lock lock(tactLock);
            tackCv.wait(lock);
        }
        switch (step) {
            case 1:
                EXPECT_TRUE(splitter->SplitterClientAdd(&clientID));
                break;
            case 5:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(4, (*outData)[0]) << "step: " << step;
                break;
            case 10:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(8, (*outData)[0]) << "step: " << step;
                break;
            case 13:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(12, (*outData)[0]) << "step: " << step;
                break;
            case 21:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(16, (*outData)[0]) << "step: " << step;
                break;
            case 36:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(32, (*outData)[0]) << "step: " << step;
                break;
            case 42:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 20);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 21) << "step: " << step;
                EXPECT_EQ(42, (*outData)[0]) << "step: " << step;
                break;
            case 45:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(44, (*outData)[0]) << "step: " << step;
                break;
            case 46:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(CLIENT_REMOVED, result_function) << "step: " << step;
                ASSERT_LE(time, 26) << "step: " << step;
                break;
            case 49:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(CLIENT_NOT_FOUND, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                return;
            default:
                break;
        }
    }
}

void MultiThreadedTest::client2_behavior() {
    int time, result_function, clientID;
    std::shared_ptr <std::vector<uint8_t>> outData;
    while (step != -1) {
        {
            std::unique_lock lock(tactLock);
            tackCv.wait(lock);
        }
        std::vector<uint8_t> data;
        switch (step) {
            case 2:
                EXPECT_TRUE(splitter->SplitterClientAdd(&clientID)) << "step: " << step;;
                break;
            case 10:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                EXPECT_EQ(time, 0) << "step: " << step;
                EXPECT_EQ(4, (*outData)[0]) << "step: " << step;
                break;
            case 13:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(8, (*outData)[0]) << "step: " << step;
                break;
            case 14:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(12, (*outData)[0]) << "step: " << step;
                break;
            case 21:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(16, (*outData)[0]) << "step: " << step;
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(20, (*outData)[0]) << "step: " << step;
                break;
            case 24:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 50) << "step: " << step;
                EXPECT_EQ(25, (*outData)[0]) << "step: " << step;
                break;
            case 28:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 70);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 70) << "step: " << step;
                EXPECT_EQ(28, (*outData)[0]) << "step: " << step;
                break;
            case 32:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 50) << "step: " << step;
                EXPECT_EQ(32, (*outData)[0]) << "step: " << step;
                break;
            case 36:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 50) << "step: " << step;
                EXPECT_EQ(33, (*outData)[0]) << "step: " << step;
                break;
            case 42:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 20);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 21) << "step: " << step;
                EXPECT_EQ(42, (*outData)[0]) << "step: " << step;
                break;
            case 45:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 50);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 1) << "step: " << step;
                EXPECT_EQ(44, (*outData)[0]) << "step: " << step;
                break;
            case 46:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 100);
                EXPECT_EQ(SUCCESS, result_function) << "step: " << step;
                ASSERT_LE(time, 76) << "step: " << step;
                EXPECT_EQ(49, (*outData)[0]) << "step: " << step;
                break;
            case 50:
                MEASURE_CALL_R(time, result_function, splitter->SplitterGet, clientID, outData, 100);
                EXPECT_EQ(CLOSED, result_function) << "step: " << step;
                ASSERT_LE(time, 26) << "step: " << step;
                return;
            default:
                break;
        }
    }
}
