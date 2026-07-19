#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;
using namespace testing;

TEST(InfoTest, ctor_null)
{
    Communicator comm;

    Info info;
    EXPECT_FALSE(info.is_valid());
}

TEST(InfoTest, set_get)
{
    Info info;
    info.set("key1", "first");
    auto k1 = info.get("key1");
    ASSERT_TRUE(k1.has_value());
    EXPECT_EQ(k1.value(), "first");

    auto k2 = info.get("no");
    ASSERT_FALSE(k2.has_value());
}

TEST(InfoTest, duplicate)
{
    Info info;
    info.set("key1", "first");

    auto dup = info.duplicate();
    auto k1 = dup.get("key1");
    ASSERT_TRUE(k1.has_value());
    EXPECT_EQ(k1.value(), "first");

    auto k2 = dup.get("no");
    ASSERT_FALSE(k2.has_value());
}

TEST(InfoTest, delete_key)
{
    Info info;
    info.set("key1", "first");
    info.set("key2", "second");
    info.del("key1");

    auto k1 = info.get("key1");
    auto k2 = info.get("key2");
    EXPECT_FALSE(k1.has_value());
    EXPECT_TRUE(k2.has_value());
}

TEST(InfoTest, get_all_keys)
{
    Info info;
    info.set("key1", "first");
    info.set("key2", "second");

    auto keys = info.keys();
    EXPECT_THAT(keys, testing::UnorderedElementsAre("key1", "key2"));
}

TEST(InfoTest, op_access)
{
    Info info;
    info.set("key1", "first");

    EXPECT_EQ(info["key1"], "first");

    EXPECT_THROW(info["none"], std::out_of_range);
}

TEST(InfoTest, get_env)
{
    {
        auto info = Info::env();
    }
    // this forced the dtor of `info` to be called, so if we get here
    // without a crash, we are good. Info::env pulls in a read-only
    // object from MPI, that should not be destroyed by the dtor.
    // So, that's why we test this.
    SUCCEED();
}
