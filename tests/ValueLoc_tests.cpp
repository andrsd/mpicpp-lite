#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <vector>

namespace mpi = mpicpp_lite;

TEST(ValueLocTest, construct_and_compare)
{
    mpi::ValueLoc<double, int> a;
    EXPECT_DOUBLE_EQ(a.value, 0.0);
    EXPECT_EQ(a.loc, 0);

    mpi::ValueLoc<double, int> b { 3.14, 2 };
    EXPECT_DOUBLE_EQ(b.value, 3.14);
    EXPECT_EQ(b.loc, 2);

    mpi::ValueLoc<double, int> c { 3.14, 2 };
    EXPECT_EQ(b, c);

    mpi::ValueLoc<double, int> d { 3.14, 1 };
    EXPECT_NE(b, d);

    // CTAD
    mpi::ValueLoc e { 10.0f, 4 };
    EXPECT_FLOAT_EQ(e.value, 10.0f);
    EXPECT_EQ(e.loc, 4);
}

TEST(ValueLocTest, minloc_functor)
{
    mpi::op::minloc oper;
    mpi::ValueLoc<double> a { 1.5, 0 };
    mpi::ValueLoc<double> b { 2.5, 1 };
    auto res1 = oper(a, b);
    EXPECT_DOUBLE_EQ(res1.value, 1.5);
    EXPECT_EQ(res1.loc, 0);

    auto res2 = oper(b, a);
    EXPECT_DOUBLE_EQ(res2.value, 1.5);
    EXPECT_EQ(res2.loc, 0);

    // Tie-break: smaller loc wins
    mpi::ValueLoc<double> c { 1.5, 3 };
    auto res3 = oper(a, c);
    EXPECT_DOUBLE_EQ(res3.value, 1.5);
    EXPECT_EQ(res3.loc, 0);

    auto res4 = oper(c, a);
    EXPECT_DOUBLE_EQ(res4.value, 1.5);
    EXPECT_EQ(res4.loc, 0);
}

TEST(ValueLocTest, maxloc_functor)
{
    mpi::op::maxloc oper;
    mpi::ValueLoc<double> a { 1.5, 0 };
    mpi::ValueLoc<double> b { 2.5, 1 };
    auto res1 = oper(a, b);
    EXPECT_DOUBLE_EQ(res1.value, 2.5);
    EXPECT_EQ(res1.loc, 1);

    auto res2 = oper(b, a);
    EXPECT_DOUBLE_EQ(res2.value, 2.5);
    EXPECT_EQ(res2.loc, 1);

    // Tie-break: smaller loc wins
    mpi::ValueLoc<double> c { 2.5, 3 };
    auto res3 = oper(b, c);
    EXPECT_DOUBLE_EQ(res3.value, 2.5);
    EXPECT_EQ(res3.loc, 1);

    auto res4 = oper(c, b);
    EXPECT_DOUBLE_EQ(res4.value, 2.5);
    EXPECT_EQ(res4.loc, 1);
}

TEST(ValueLocTest, all_reduce_minloc_double)
{
    mpi::Communicator comm;

    mpi::ValueLoc<double> loc { static_cast<double>(comm.rank() + 10), comm.rank() };
    mpi::ValueLoc<double> glob;

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    EXPECT_DOUBLE_EQ(glob.value, 10.0);
    EXPECT_EQ(glob.loc, 0);

    // In-place
    comm.all_reduce(loc, mpi::op::minloc<>());
    EXPECT_DOUBLE_EQ(loc.value, 10.0);
    EXPECT_EQ(loc.loc, 0);
}

TEST(ValueLocTest, all_reduce_maxloc_double)
{
    mpi::Communicator comm;

    mpi::ValueLoc<double> loc { static_cast<double>(comm.rank() + 10), comm.rank() };
    mpi::ValueLoc<double> glob;

    comm.all_reduce(loc, glob, mpi::op::maxloc<>());
    int max_rank = comm.size() - 1;
    EXPECT_DOUBLE_EQ(glob.value, static_cast<double>(max_rank + 10));
    EXPECT_EQ(glob.loc, max_rank);

    // In-place
    comm.all_reduce(loc, mpi::op::maxloc<>());
    EXPECT_DOUBLE_EQ(loc.value, static_cast<double>(max_rank + 10));
    EXPECT_EQ(loc.loc, max_rank);
}

TEST(ValueLocTest, all_reduce_minloc_int)
{
    mpi::Communicator comm;

    mpi::ValueLoc<int> loc { (comm.rank() + 5) * 2, comm.rank() };
    mpi::ValueLoc<int> glob;

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    EXPECT_EQ(glob.value, 10);
    EXPECT_EQ(glob.loc, 0);
}

TEST(ValueLocTest, all_reduce_maxloc_int)
{
    mpi::Communicator comm;

    mpi::ValueLoc<int> loc { (comm.rank() + 5) * 2, comm.rank() };
    mpi::ValueLoc<int> glob;

    comm.all_reduce(loc, glob, mpi::op::maxloc<>());
    int max_rank = comm.size() - 1;
    EXPECT_EQ(glob.value, (max_rank + 5) * 2);
    EXPECT_EQ(glob.loc, max_rank);
}

TEST(ValueLocTest, all_reduce_minloc_float)
{
    mpi::Communicator comm;

    mpi::ValueLoc<float> loc { 100.0f - static_cast<float>(comm.rank()), comm.rank() };
    mpi::ValueLoc<float> glob;

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    int max_rank = comm.size() - 1;
    EXPECT_FLOAT_EQ(glob.value, 100.0f - static_cast<float>(max_rank));
    EXPECT_EQ(glob.loc, max_rank);
}

TEST(ValueLocTest, all_reduce_minloc_short)
{
    mpi::Communicator comm;

    mpi::ValueLoc<short> loc { static_cast<short>(comm.rank() + 1), comm.rank() };
    mpi::ValueLoc<short> glob;

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    EXPECT_EQ(glob.value, 1);
    EXPECT_EQ(glob.loc, 0);
}

TEST(ValueLocTest, all_reduce_minloc_long)
{
    mpi::Communicator comm;

    mpi::ValueLoc<long> loc { static_cast<long>(comm.rank() + 1), comm.rank() };
    mpi::ValueLoc<long> glob;

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    EXPECT_EQ(glob.value, 1);
    EXPECT_EQ(glob.loc, 0);
}

TEST(ValueLocTest, all_reduce_tie_breaking)
{
    mpi::Communicator comm;

    // All ranks provide the exact same value but different rank loc
    mpi::ValueLoc<double> loc { 42.0, comm.rank() };
    mpi::ValueLoc<double> glob_min;
    mpi::ValueLoc<double> glob_max;

    comm.all_reduce(loc, glob_min, mpi::op::minloc<>());
    EXPECT_DOUBLE_EQ(glob_min.value, 42.0);
    EXPECT_EQ(glob_min.loc, 0);

    comm.all_reduce(loc, glob_max, mpi::op::maxloc<>());
    EXPECT_DOUBLE_EQ(glob_max.value, 42.0);
    EXPECT_EQ(glob_max.loc, 0);
}

TEST(ValueLocTest, reduce_minloc_root)
{
    mpi::Communicator comm;

    mpi::ValueLoc<double> loc { static_cast<double>(comm.rank() + 7), comm.rank() };
    mpi::ValueLoc<double> glob;

    comm.reduce(loc, glob, mpi::op::minloc<>(), 0);
    if (comm.rank() == 0) {
        EXPECT_DOUBLE_EQ(glob.value, 7.0);
        EXPECT_EQ(glob.loc, 0);
    }
}

TEST(ValueLocTest, reduce_maxloc_root)
{
    mpi::Communicator comm;

    mpi::ValueLoc<double> loc { static_cast<double>(comm.rank() + 7), comm.rank() };
    mpi::ValueLoc<double> glob;

    comm.reduce(loc, glob, mpi::op::maxloc<>(), 0);
    if (comm.rank() == 0) {
        int max_rank = comm.size() - 1;
        EXPECT_DOUBLE_EQ(glob.value, static_cast<double>(max_rank + 7));
        EXPECT_EQ(glob.loc, max_rank);
    }
}

TEST(ValueLocTest, all_reduce_vector)
{
    mpi::Communicator comm;

    std::vector<mpi::ValueLoc<int>> loc = {
        { comm.rank() + 10, comm.rank() },
        { 100 - comm.rank(), comm.rank() }
    };
    std::vector<mpi::ValueLoc<int>> glob(2);

    comm.all_reduce(loc, glob, mpi::op::minloc<>());
    EXPECT_EQ(glob[0].value, 10);
    EXPECT_EQ(glob[0].loc, 0);

    int max_rank = comm.size() - 1;
    EXPECT_EQ(glob[1].value, 100 - max_rank);
    EXPECT_EQ(glob[1].loc, max_rank);
}
