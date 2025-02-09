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
    auto win = Window::create(arr, MPI_INFO_NULL, comm);

    int dest = (comm.rank() + 1) % comm.size();

    win.lock(Lock::SHARED, dest);
    win.get(remote_arr.data(), n, dest, 0, n);
    win.unlock(dest);

    for (int i = 0; i < n; i++)
        EXPECT_EQ(remote_arr[i], (dest * 10) + i);
}
