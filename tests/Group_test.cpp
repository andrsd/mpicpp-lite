#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;
using namespace testing;

TEST(GroupTest, include)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    int rank = comm.rank();

    auto world_group = comm.group();
    auto sub_group = world_group.include({ 0, 2, 3 });
    EXPECT_EQ(sub_group.size(), 3);
    if (rank == 0)
        EXPECT_EQ(sub_group.rank(), 0);
    else if (rank == 1)
        EXPECT_EQ(sub_group.rank(), UNDEFINED);
    else if (rank == 2)
        EXPECT_EQ(sub_group.rank(), 1);
    else if (rank == 3)
        EXPECT_EQ(sub_group.rank(), 2);

    auto sub_comm = comm.create(sub_group);
    if (sub_comm.is_valid()) {
        EXPECT_EQ(sub_comm.size(), 3);
        if (rank == 0)
            EXPECT_EQ(sub_comm.rank(), 0);
        else if (rank == 2)
            EXPECT_EQ(sub_comm.rank(), 1);
        else if (rank == 3)
            EXPECT_EQ(sub_comm.rank(), 2);
    }

    sub_group.free();
    world_group.free();
}

TEST(GroupTest, exclude)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    int rank = comm.rank();

    auto world_group = comm.group();
    auto sub_group = world_group.exclude({ 0, 3 });
    EXPECT_EQ(sub_group.size(), 2);
    if (rank == 0)
        EXPECT_EQ(sub_group.rank(), UNDEFINED);
    else if (rank == 1)
        EXPECT_EQ(sub_group.rank(), 0);
    else if (rank == 2)
        EXPECT_EQ(sub_group.rank(), 1);
    else if (rank == 3)
        EXPECT_EQ(sub_group.rank(), UNDEFINED);

    auto sub_comm = comm.create(sub_group);
    if (sub_comm.is_valid()) {
        EXPECT_EQ(sub_comm.size(), 2);
        if (rank == 1)
            EXPECT_EQ(sub_comm.rank(), 0);
        else if (rank == 2)
            EXPECT_EQ(sub_comm.rank(), 1);
    }

    sub_group.free();
    world_group.free();
}

TEST(GroupTest, translate_rank)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();
    auto sub_group = world_group.include({ 1, 3 });

    auto rank0 = world_group.translate_rank(0, sub_group);
    EXPECT_EQ(rank0, UNDEFINED);
    auto rank1 = world_group.translate_rank(1, sub_group);
    EXPECT_EQ(rank1, 0);
    auto rank2 = world_group.translate_rank(2, sub_group);
    EXPECT_EQ(rank2, UNDEFINED);
    auto rank3 = world_group.translate_rank(3, sub_group);
    EXPECT_EQ(rank3, 1);
}

TEST(GroupTest, translate_ranks)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();
    auto sub_group = world_group.include({ 1, 3 });
    auto dest_ranks = world_group.translate_ranks({ 1, 2, 3 }, sub_group);
    EXPECT_THAT(dest_ranks, ElementsAre(0, UNDEFINED, 1));
}

TEST(GroupTest, compare)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();

    auto one = world_group.include({ 0, 2 });
    auto two = world_group.include({ 0, 2 });
    EXPECT_EQ(Group::compare(one, two), Group::IDENTICAL);

    auto two2 = world_group.include({ 2, 0 });
    EXPECT_EQ(Group::compare(one, two2), Group::SIMILAR);

    auto three = world_group.include({ 3 });
    EXPECT_EQ(Group::compare(one, three), Group::UNEQUAL);

    three.free();
    two2.free();
    two.free();
    one.free();
    world_group.free();
}

TEST(GroupTest, group_union)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();

    auto one = world_group.include({ 0, 1 });
    auto two = world_group.include({ 1, 3 });
    auto un = Group::join(one, two);
    EXPECT_EQ(un.size(), 3);
    auto un_gold = world_group.include({ 0, 1, 3 });
    EXPECT_EQ(Group::compare(un, un_gold), Group::IDENTICAL);

    un.free();
    un_gold.free();
    two.free();
    one.free();
    world_group.free();
}

TEST(GroupTest, group_intersection)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();

    auto one = world_group.include({ 0, 1 });
    auto two = world_group.include({ 1, 3 });
    auto un = Group::intersection(one, two);
    EXPECT_EQ(un.size(), 1);
    auto un_gold = world_group.include({ 1 });
    EXPECT_EQ(Group::compare(un, un_gold), Group::IDENTICAL);

    un.free();
    un_gold.free();
    two.free();
    one.free();
    world_group.free();
}

TEST(GroupTest, group_difference)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    auto world_group = comm.group();

    auto one = world_group.include({ 0, 1 });
    auto two = world_group.include({ 1, 3 });
    auto un = Group::difference(one, two);
    EXPECT_EQ(un.size(), 1);
    auto un_gold = world_group.include({ 0 });
    EXPECT_EQ(Group::compare(un, un_gold), Group::IDENTICAL);

    un.free();
    un_gold.free();
    two.free();
    one.free();
    world_group.free();
}
