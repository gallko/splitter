#include <ISplitter.h>
#include <gtest/gtest.h>
#include <chrono>

#define MEASURE_CALL(time, function, ...)  { \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    function(__VA_ARGS__);                                     \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count(); }

#define MEASURE_CALL_R(time, result_function, function, ...)    { \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    result_function = function(__VA_ARGS__);                   \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() ; }

class SingleThreadTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        splitter_0_10 = SplitterCreate(0, 10);
        splitter_1_5 = SplitterCreate(1, 5);
        splitter_5_15 = SplitterCreate(5, 15);
    }

    std::shared_ptr<ISplitter> splitter_0_10;
    std::shared_ptr<ISplitter> splitter_1_5;
    std::shared_ptr<ISplitter> splitter_5_15;
};

TEST_F(SingleThreadTest, SplitterInfoGet) {
    int nMaxBuffers = -1, nMaxClients = -1;
    splitter_0_10->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 0);
    EXPECT_EQ(nMaxClients, 10);
    splitter_1_5->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 1);
    EXPECT_EQ(nMaxClients, 5);
    splitter_5_15->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 5);
    EXPECT_EQ(nMaxClients, 15);
}

TEST_F(SingleThreadTest, SplitterClientAdd) {
    int clientID = 0, count = -1;
    std::map<int, int> idClients;

    EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));
    EXPECT_EQ(0, count);

    for(int i = 0; i < 10; ++i) {
        EXPECT_TRUE(splitter_0_10->SplitterClientAdd(&clientID));
        EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));
        EXPECT_EQ(i + 1, count);
        idClients[clientID] += 1;
    }

    EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));
    EXPECT_EQ(10, count);
    EXPECT_EQ(count, idClients.size());

    EXPECT_FALSE(splitter_0_10->SplitterClientAdd(&clientID));
    EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));

    EXPECT_EQ(10, count);
}

TEST_F(SingleThreadTest, SplitterClientRemove) {
    constexpr int countClient = 5;
    int count = -1;
    std::vector<int> idClients;
    idClients.resize(countClient);
    for(int i = 0; i < countClient; ++i) {
        EXPECT_TRUE(splitter_0_10->SplitterClientAdd(&idClients[i]));
    }
    auto tmpClientID = idClients[0];

    EXPECT_TRUE(splitter_0_10->SplitterClientRemove(idClients[3]));
    EXPECT_TRUE(splitter_0_10->SplitterClientRemove(idClients[0]));

    EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));
    EXPECT_EQ(3, count);

    EXPECT_TRUE(splitter_0_10->SplitterClientRemove(idClients[1]));
    EXPECT_TRUE(splitter_0_10->SplitterClientRemove(idClients[2]));
    EXPECT_TRUE(splitter_0_10->SplitterClientRemove(idClients[4]));

    EXPECT_TRUE(splitter_0_10->SplitterClientGetCount(&count));
    EXPECT_EQ(0, count);

    EXPECT_FALSE(splitter_0_10->SplitterClientRemove(tmpClientID));

    srand( time( 0 ) );
    EXPECT_FALSE(splitter_0_10->SplitterClientRemove(rand()));
}

TEST_F(SingleThreadTest, SplitterClientGetCount) {
    EXPECT_TRUE(true) << "checked in SplitterClientAdd and SplitterClientRemove";
}

TEST_F(SingleThreadTest, SplitterClientGetByIndex) {
    EXPECT_TRUE(true) << "checked in SplitterPut, SplitterGet and SplitterFlush";
}

TEST_F(SingleThreadTest, SplitterPut) {
    int clientID0 = 0, clientID1 = 0, clientID = -1, latency = -1;

    EXPECT_FALSE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_FALSE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));

    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID0));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID1));

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(0, latency);

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(0, latency);

    EXPECT_FALSE(splitter_5_15->SplitterClientGetByIndex(2, &clientID, &latency));

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterPut(nullptr, i * 10));
        EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
        EXPECT_EQ(i + 1, latency);
    }

    latency = -1;
    int time, result_function;
    MEASURE_CALL_R(time, result_function, splitter_5_15->SplitterPut, nullptr, 50);
    ASSERT_GE(time, 50); ASSERT_LE(time, 51);
    EXPECT_EQ(CLIENT_MISSED_DATA, result_function);

    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(5, latency);

    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(5, latency);

}

TEST_F(SingleThreadTest, SplitterFlush) {
    int clientID0 = 0, clientID1 = 0, clientID = -1, latency = -1;

    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID0));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID1));

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(0, latency);

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(0, latency);

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterPut(nullptr, i * 10));
        EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
        EXPECT_EQ(i + 1, latency);
    }

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(4, latency);

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(4, latency);

    EXPECT_EQ(SUCCESS, splitter_5_15->SplitterFlush());

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(0, latency);

    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(0, latency);
}

TEST_F(SingleThreadTest, SplitterGet) {
    int clientID0 = 0, clientID1 = 0, clientID2 = 0, clientID = 0, latency = -1;
    int time, result_function;
    std::shared_ptr <std::vector<uint8_t>> outData;

    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID0));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID1));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID2));

    MEASURE_CALL_R(time, result_function, splitter_5_15->SplitterGet, clientID0, outData, 60);
    ASSERT_GE(time, 60); ASSERT_LE(time, 61);
    EXPECT_EQ(TIME_OUT, result_function);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterPut(std::make_shared<std::vector<uint8_t>>(i + 1), i * 10));
    }

    /* clientID0 */
    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(clientID, clientID0);
    EXPECT_EQ(5, latency);
    for (int i = 0; i < 3; ++i) {
        MEASURE_CALL_R(time, result_function, splitter_5_15->SplitterGet, clientID0, outData, 50);
        EXPECT_EQ(SUCCESS, result_function);
        EXPECT_EQ(i + 1, outData->size());
        ASSERT_LE(time, 1);
    }

    /* clientID1 */
    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(clientID, clientID1);
    EXPECT_EQ(5, latency);
    for (int i = 0; i < 5; ++i) {
        MEASURE_CALL_R(time, result_function, splitter_5_15->SplitterGet, clientID1, outData, 50);
        EXPECT_EQ(SUCCESS, result_function);
        EXPECT_EQ(i + 1, outData->size());
        ASSERT_LE(time, 1);
    }

    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(0, &clientID, &latency));
    EXPECT_EQ(clientID, clientID0);
    EXPECT_EQ(2, latency);

    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(1, &clientID, &latency));
    EXPECT_EQ(clientID, clientID1);
    EXPECT_EQ(0, latency);

    latency = -1;
    EXPECT_TRUE(splitter_5_15->SplitterClientGetByIndex(2, &clientID, &latency));
    EXPECT_EQ(clientID, clientID2);
    EXPECT_EQ(5, latency);
}

TEST_F(SingleThreadTest, SplitterClose) {
    int clientID0 = 0, clientID1 = 0, clientID2 = 0, latency = 0;
    std::shared_ptr <std::vector<uint8_t>> outData;

    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID0));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID1));
    EXPECT_TRUE(splitter_5_15->SplitterClientAdd(&clientID2));

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterPut(nullptr, 50));
    }

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterGet(clientID0, outData, 50));
    }

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(SUCCESS, splitter_5_15->SplitterGet(clientID1, outData, 50));
    }
    splitter_5_15->SplitterClose();

    EXPECT_FALSE(splitter_5_15->SplitterInfoGet(&clientID0, &clientID1));
    EXPECT_EQ(CLOSED, splitter_5_15->SplitterPut(nullptr, 50));
    EXPECT_EQ(CLOSED, splitter_5_15->SplitterFlush());
    EXPECT_FALSE(splitter_5_15->SplitterClientAdd(&clientID0));
    EXPECT_FALSE( splitter_5_15->SplitterClientRemove(clientID0));
    EXPECT_FALSE( splitter_5_15->SplitterClientGetCount(&clientID0));
    EXPECT_FALSE( splitter_5_15->SplitterClientGetByIndex(0, &clientID0, &latency));
    EXPECT_EQ(CLOSED, splitter_5_15->SplitterGet(clientID0, outData, 50));

    splitter_5_15->SplitterClose();
}
