#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;

TEST(ExceptionTest, exception)
{
    Communicator comm;
    if (comm.size() < 2)
        return;

    int faulty_rank = -1;
    {
        CollectiveExceptionHandler handler(comm);
        try {
            if (comm.rank() == 1) {
                throw std::runtime_error("my error");
            }

            handler.sync();
        }
        catch (std::exception & e) {
            auto par_exc = handler.sync(e);
            //
            faulty_rank = par_exc.rank();
            par_exc.broadcast(comm);

            if (comm.rank() == faulty_rank) {
                EXPECT_STREQ(par_exc.what(), "my error");
            }
        }
    }

    comm.barrier();
    EXPECT_EQ(faulty_rank, 1);
}
