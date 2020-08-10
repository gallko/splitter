#include <ISplitter.h>
#include <gtest/gtest.h>
#include <chrono>

#define MEASURE_CALL(time, function, ...) \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    function(__VA_ARGS__);                                     \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

#define MEASURE_CALL_R(time, result_function, function, ...)     \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    result_function = function(__VA_ARGS__);                   \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

class SingleThreadTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        fifo0 = SplitterCreate(0, 10);
        fifo1 = SplitterCreate(3, 5);
        fifo2 = SplitterCreate(5, 15);
    }

    std::shared_ptr<ISplitter> fifo0;
    std::shared_ptr<ISplitter> fifo1;
    std::shared_ptr<ISplitter> fifo2;
};

TEST_F(SingleThreadTest, IsEmptyInitially) {
    int nMaxBuffers = -1, nMaxClients = -1;
    fifo0->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 0);
    EXPECT_EQ(nMaxClients, 10);

    fifo1->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 3);
    EXPECT_EQ(nMaxClients, 5);

    fifo2->SplitterInfoGet(&nMaxBuffers, &nMaxClients);
    EXPECT_EQ(nMaxBuffers, 5);
    EXPECT_EQ(nMaxClients, 15);
}

TEST_F(SingleThreadTest, AddingClient) {
    int clientID0 = 0, clientID1 = 0, clientID2 = 0, count = -1;
    bool result;

    result = fifo0->SplitterClientGetCount(&count);
    EXPECT_EQ(0, count);
    EXPECT_TRUE(result);

    result = fifo0->SplitterClientAdd(&clientID0);
    EXPECT_TRUE(result);
    result = fifo0->SplitterClientAdd(&clientID1);
    EXPECT_TRUE(result);
    result = fifo0->SplitterClientAdd(&clientID2);
    EXPECT_TRUE(result);

    ASSERT_NE(0, clientID0);
    ASSERT_NE(0, clientID1);
    ASSERT_NE(0, clientID2);
    ASSERT_NE(clientID0, clientID1);
    ASSERT_NE(clientID0, clientID2);
    ASSERT_NE(clientID1, clientID2);

    for(int i = 0; i < 7; ++i) {
        result = fifo0->SplitterClientAdd(&count);
        EXPECT_TRUE(result);
        result = fifo0->SplitterClientGetCount(&count);
        EXPECT_TRUE(result);
        EXPECT_EQ(i + 4, count);
    }

    result = fifo0->SplitterClientGetCount(&count);
    EXPECT_EQ(10, count);
    EXPECT_TRUE(result);

    result = fifo0->SplitterClientAdd(&count);
    EXPECT_FALSE(result);
    EXPECT_EQ(10, count);
}

TEST_F(SingleThreadTest, ExtractDataFromEmpty) {

}

TEST_F(SingleThreadTest, OverflowQueue) {

}