#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;

TEST(CartesianCommunicator, create)
{
    Communicator comm;
    if (comm.size() != 6)
        return;

    auto cart_comm = comm.create_cartesian({ 2, 3 });

    EXPECT_EQ(cart_comm.dimensions(), 2);
}

TEST(CartesianCommunicator, rank)
{
    Communicator comm;
    if (comm.size() != 6)
        return;

    auto cart_comm = comm.create_cartesian({ 2, 3 });
    EXPECT_EQ(cart_comm.rank(0, 0), 0);
    EXPECT_EQ(cart_comm.rank(0, 1), 1);
    EXPECT_EQ(cart_comm.rank(0, 2), 2);
    EXPECT_EQ(cart_comm.rank(1, 0), 3);
    EXPECT_EQ(cart_comm.rank(1, 1), 4);
    EXPECT_EQ(cart_comm.rank(1, 2), 5);
}

TEST(CartesianCommunicator, coords)
{
    Communicator comm;
    if (comm.size() != 6)
        return;

    auto cart_comm = comm.create_cartesian({ 2, 3 });
    EXPECT_THAT(cart_comm.coords(0), testing::ElementsAre(0, 0));
    EXPECT_THAT(cart_comm.coords(1), testing::ElementsAre(0, 1));
    EXPECT_THAT(cart_comm.coords(2), testing::ElementsAre(0, 2));
    EXPECT_THAT(cart_comm.coords(3), testing::ElementsAre(1, 0));
    EXPECT_THAT(cart_comm.coords(4), testing::ElementsAre(1, 1));
    EXPECT_THAT(cart_comm.coords(5), testing::ElementsAre(1, 2));
}

TEST(CartesianCommunicator, create_6)
{
    Communicator comm;
    if (comm.size() != 6)
        return;

    auto dims = create_dims(comm.size(), 2);
    EXPECT_EQ(dims[0], 3);
    EXPECT_EQ(dims[1], 2);
}
