#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;
using namespace testing;

TEST(WindowTest, get)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    constexpr int n = 10;
    std::vector<int> arr(n, 0.);
    for (int i = 0; i < n; i++)
        arr[i] = (comm.rank() * 10) + i;

    std::vector<int> remote_arr(n, 0.);
    auto win = Window::create(std::span { arr }, Info {}, comm);

    int dest = (comm.rank() + 1) % comm.size();

    win.lock(Lock::SHARED, dest);
    win.get(remote_arr.data(), n, dest, 0, n);
    win.unlock(dest);

    for (int i = 0; i < n; i++)
        EXPECT_EQ(remote_arr[i], (dest * 10) + i);

    win.free();
}

TEST(WindowTest, allocate)
{
    Communicator comm;
    constexpr int n = 5;
    auto [win, base] = Window::allocate<int>(n, Info {}, comm);
    EXPECT_TRUE(win);
    EXPECT_NE(base, nullptr);

    auto base_attr = win.attr<void *>(win_base);
    ASSERT_TRUE(base_attr.has_value());
    EXPECT_EQ(base_attr.value(), base);

    auto size_attr = win.attr<MPI_Aint>(win_size);
    ASSERT_TRUE(size_attr.has_value());
    EXPECT_EQ(size_attr.value(), n * sizeof(int));

    win.free();
}

TEST(WindowTest, info)
{
    Communicator comm;
    auto [win, base] = Window::allocate<int>(5, Info {}, comm);

    Info info;
    info.set("no_locks", "true");
    win.set_info(info);

    auto info_back = win.info();
    EXPECT_TRUE(info_back.is_valid());

    win.free();
}

TEST(WindowTest, custom_attributes)
{
    Communicator comm;
    auto [win, base] = Window::allocate<int>(5, Info {}, comm);

    auto key = Window::create_key();
    int val = 9876;
    win.set_attr(key, val);

    auto val_back = win.attr<int>(key);
    ASSERT_TRUE(val_back.has_value());
    EXPECT_EQ(val_back.value(), 9876);

    win.delete_attr(key);
    auto val_deleted = win.attr<int>(key);
    ASSERT_FALSE(val_deleted.has_value());

    win.free();
}

TEST(WindowTest, fence)
{
    Communicator comm;
    auto [win, base] = Window::allocate<int>(5, Info {}, comm);

    win.fence();
    win.fence(MPI_MODE_NOPRECEDE);
    win.fence(MPI_MODE_NOSUCCEED);

    win.free();
}
