#include "gtest/gtest.h"
#include "mpicpp-lite/mpicpp-lite.h"

namespace mpi = mpicpp_lite;

int
main(int argc, char ** argv)
{
    mpi::Environment env(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    return RUN_ALL_TESTS();
}
