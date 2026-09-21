#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

namespace mpi = mpicpp_lite;

TEST(OperationsTest, sum)
{
    double a = 10;
    double b = 21;
    auto oper = mpi::op::sum<>();
    EXPECT_DOUBLE_EQ(oper(a, b), 31.);
}

TEST(OperationsTest, prod)
{
    double a = 10;
    double b = 21;
    auto oper = mpi::op::prod<>();
    EXPECT_DOUBLE_EQ(oper(a, b), 210.);
}

TEST(OperationsTest, max)
{
    double a = 10;
    double b = 21;
    auto oper = mpi::op::max<>();
    EXPECT_DOUBLE_EQ(oper(a, b), 21.);
}

TEST(OperationsTest, min)
{
    double a = 10;
    double b = 21;
    auto oper = mpi::op::min<>();
    EXPECT_DOUBLE_EQ(oper(a, b), 10.);
}

TEST(OperationsTest, logical_and)
{
    bool a = true;
    bool b = false;
    auto oper = mpi::op::logical_and<>();
    EXPECT_FALSE(oper(a, b));
}

TEST(OperationsTest, logical_or)
{
    bool a = true;
    bool b = false;
    auto oper = mpi::op::logical_or<>();
    EXPECT_TRUE(oper(a, b));
}

TEST(OperationsTest, logical_xor)
{
    bool a = true;
    bool b = true;
    auto oper = mpi::op::logical_xor<>();
    EXPECT_FALSE(oper(a, b));
}

TEST(OperationsTest, replace)
{
    double a = 10;
    double b = 13;
    auto oper = mpi::op::replace<>();
    EXPECT_DOUBLE_EQ(oper(a, b), 10);
}
