#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;
using namespace testing;

TEST(EnvironmentTest, basic)
{
    EXPECT_TRUE(mpicpp_lite::Environment::is_initialized());
    EXPECT_FALSE(mpicpp_lite::Environment::is_finalized());
    EXPECT_TRUE(mpicpp_lite::Environment::is_thread_main());
    auto thread_level = mpicpp_lite::Environment::query_thread();
    EXPECT_TRUE(thread_level == ThreadSupport::SINGLE || thread_level == ThreadSupport::FUNNELED ||
                thread_level == ThreadSupport::SERIALIZED ||
                thread_level == ThreadSupport::MULTIPLE);
}

TEST(EnvironmentTest, processor_name)
{
    auto name = mpicpp_lite::processor_name();
    EXPECT_FALSE(name.empty());
}
