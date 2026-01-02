#include <gmock/gmock.h>
#include "mpicpp-lite/mpicpp-lite.h"
#include <thread>
#include <chrono>

using namespace mpicpp_lite;
using namespace testing;

TEST(TimeTest, wall_time)
{
    Communicator comm;

    comm.barrier();
    auto t0 = wall_time();
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(200ms);
    auto t1 = wall_time();
    EXPECT_GT(t1 - t0, 0.2);
}
