#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;
using namespace testing;

TEST(MPITest, error)
{
    Communicator comm;
    if (comm.size() > 1)
        return;

    comm.set_error_handler();
    auto ierr = MPI_Bcast(nullptr, 0, MPI_INT, -1, MPI_COMM_WORLD);
    EXPECT_DEATH(mpicpp_lite::internal::check_mpi_error(comm, ierr, "myfile.cpp", 123),
                 "\\[ERROR\\] MPI error [0-9]+ at myfile.cpp:123:.+[Ii]nvalid root");
}

TEST(MPITest, error_message)
{
    Communicator comm;
    comm.set_error_handler();
    auto ierr = MPI_Bcast(nullptr, 0, MPI_INT, -1, MPI_COMM_WORLD);
    auto msg = error_message(error_class(ierr));
    EXPECT_THAT(msg, MatchesRegex(".*[Ii]nvalid root"));
}

TEST(MPITest, abort)
{
    Communicator comm;
    if (comm.size() > 1)
        return;

    EXPECT_DEATH(comm.abort(255), "");
}

TEST(MPITest, size_rank)
{
    int sz, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Communicator comm;
    EXPECT_EQ(comm.size(), sz);
    EXPECT_EQ(comm.rank(), rank);
}

TEST(MPITest, barrier)
{
    Communicator comm;
    comm.barrier();
}

TEST(MPITest, send_recv_int)
{
    Communicator comm;
    int n_mpis = comm.size();
    if (n_mpis == 1)
        return;

    int tag = 1234;
    if (comm.rank() == 0) {
        for (int i = 1; i < n_mpis; i++) {
            int val;
            auto status = comm.recv(i, tag, val);
            EXPECT_EQ(val, i * 4);
            EXPECT_EQ(status.tag(), tag);
            EXPECT_EQ(status.source(), i);
            EXPECT_EQ(status.error(), 0);
        }
    }
    else {
        int number = 4 * comm.rank();
        comm.send(0, tag, number);
    }
}

TEST(MPITest, send_recv_bool)
{
    Communicator comm;
    int n_mpis = comm.size();
    if (n_mpis == 1)
        return;

    int tag = 1234;
    if (comm.rank() == 0) {
        for (int i = 1; i < n_mpis; i++) {
            bool val;
            auto status = comm.recv(i, tag, val);
            EXPECT_EQ(val, i % 2 == 0);
            EXPECT_EQ(status.tag(), tag);
            EXPECT_EQ(status.source(), i);
            EXPECT_EQ(status.error(), 0);
        }
    }
    else {
        bool val = comm.rank() % 2 == 0;
        comm.send(0, tag, val);
    }
}

TEST(MPITest, send_recv)
{
    Communicator comm;
    int n_mpis = comm.size();
    if (n_mpis == 1)
        return;

    int tag = 0;
    if (comm.rank() == 0) {
        for (int i = 1; i < n_mpis; i++) {
            auto status = comm.recv(i, tag);
            EXPECT_EQ(status.source(), i);
            EXPECT_EQ(status.tag(), tag);
        }
    }
    else
        comm.send(0, tag);
}

TEST(MPITest, send_recv_arr_int)
{
    Communicator comm;
    int n_mpis = comm.size();
    if (n_mpis == 1)
        return;

    int tag = 0;
    if (comm.rank() == 0) {
        for (int i = 1; i < n_mpis; i++) {
            std::vector<int> arr;
            auto status = comm.recv(i, tag, arr);
            auto sz = arr.size();
            EXPECT_EQ(sz, i);
            for (int j = 0; j < sz; j++)
                EXPECT_EQ(arr[j], 2 * j);
            EXPECT_EQ(status.source(), i);
            EXPECT_EQ(status.tag(), tag);
        }
    }
    else {
        int n = comm.rank();
        std::vector<int> arr;
        arr.resize(n);
        for (int i = 0; i < n; i++)
            arr[i] = 2 * i;
        comm.send(0, tag, arr);
    }
}

TEST(MPITest, broadcast)
{
    Communicator comm;

    int number = 0;
    if (comm.rank() == 0)
        number = 1234;
    comm.broadcast(number, 0);
    EXPECT_EQ(number, 1234);
}

TEST(MPITest, broadcast_arr)
{
    Communicator comm;

    std::vector<int> nums;
    nums.resize(10);
    if (comm.rank() == 0) {
        for (int i = 0; i < 10; i++)
            nums[i] = 3 * i;
    }
    comm.broadcast(nums.data(), 10, 0);
    for (std::size_t i = 0; i < 10; i++)
        EXPECT_EQ(nums[i], 3 * i);
}

TEST(MPITest, broadcast_std_vector)
{
    Communicator comm;

    std::vector<int> nums;
    if (comm.rank() == 0) {
        nums.resize(10);
        for (int i = 0; i < 10; i++)
            nums[i] = 3 * i;
    }
    comm.broadcast(nums, 0);
    for (std::size_t i = 0; i < 10; i++)
        EXPECT_EQ(nums[i], 3 * i);
}

TEST(MPITest, broadcast_str)
{
    Communicator comm;

    std::string str;
    if (comm.rank() == 0) {
        str = "text to bcast";
    }
    comm.broadcast(str, 0);
    EXPECT_EQ(str, "text to bcast");
}

TEST(MPITest, gather)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int number = comm.rank() * 5;
    std::vector<int> vals;
    comm.gather(number, vals, 0);
    if (comm.rank() == 0) {
        for (std::size_t i = 0; i < comm.size(); i++)
            EXPECT_EQ(vals[i], i * 5);
    }
}

TEST(MPITest, gather_n)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int number[2] = { comm.rank() * 5, comm.rank() * 7 };
    std::vector<int> vals;
    comm.gather(number, 2, vals, 0);
    if (comm.rank() == 0) {
        for (std::size_t i = 0; i < comm.size(); i++) {
            EXPECT_EQ(vals[2 * i], i * 5);
            EXPECT_EQ(vals[2 * i + 1], i * 7);
        }
    }
}

TEST(MPITest, all_gather_1)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int val = comm.rank() * 5;

    std::vector<int> all_values;
    comm.all_gather(val, all_values);

    EXPECT_EQ(all_values.size(), comm.size());
    for (int i = 0; i < comm.size(); i++)
        EXPECT_EQ(all_values[i], i * 5);
}

TEST(MPITest, all_gather_vec_1_proc)
{
    Communicator comm;
    if (comm.size() != 1)
        return;

    std::vector<int> vals = { 3, 2, 6 };
    std::vector<int> all_vals;
    comm.all_gather(vals, all_vals);
    EXPECT_EQ(all_vals.size(), 3);
    EXPECT_THAT(all_vals, ElementsAre(3, 2, 6));
}

TEST(MPITest, all_gather_vec_4_procs)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    std::vector<int> vals;
    if (comm.rank() == 0)
        vals = { 1, 3 };
    else if (comm.rank() == 1)
        vals = { 0, 2, 4 };
    else if (comm.rank() == 2)
        vals = { 5 };
    else if (comm.rank() == 3)
        vals = {};

    std::vector<int> all_vals;
    comm.all_gather(vals, all_vals);
    EXPECT_EQ(all_vals.size(), 6);
    EXPECT_THAT(all_vals, ElementsAre(1, 3, 0, 2, 4, 5));
}

TEST(MPITest, scatter)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    std::vector<int> vals;
    if (comm.rank() == 0) {
        vals.resize(comm.size());
        for (int i = 0; i < comm.size(); i++)
            vals[i] = (i + 1) * 5;
    }
    int number = -1;
    comm.scatter(vals, number, 0);

    EXPECT_EQ(number, (comm.rank() + 1) * 5);
}

TEST(MPITest, scatter_2)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    std::vector<int> vals;
    if (comm.rank() == 0) {
        vals.resize(comm.size());
        for (int i = 0; i < comm.size(); i++)
            vals[i] = (i + 1) * 5;
    }
    int number = -1;
    comm.scatter(vals.data(), number, 0);

    EXPECT_EQ(number, (comm.rank() + 1) * 5);
}

TEST(MPITest, scatter_n)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    std::vector<int> vals;
    if (comm.rank() == 0) {
        vals.resize(comm.size() * 2);
        for (int i = 0; i < comm.size(); i++) {
            vals[2 * i] = (i + 1) * 5;
            vals[2 * i + 1] = (i + 1) * 7;
        }
    }
    int number[2] = { -1, -1 };
    comm.scatter(vals, number, 2, 0);

    EXPECT_EQ(number[0], (comm.rank() + 1) * 5);
    EXPECT_EQ(number[1], (comm.rank() + 1) * 7);
}

TEST(MPITest, reduce_sum)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int n = comm.size();
    double loc_sum = (comm.rank() + 1) * 3;
    double glob_sum;
    comm.reduce(loc_sum, glob_sum, op::sum<double>(), 0);

    if (comm.rank() == 0) {
        double gold = 3. * (n * (1 + n) / 2.);
        EXPECT_EQ(glob_sum, gold);
    }
}

TEST(MPITest, reduce_sum_arr)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int n = comm.size();
    std::vector<int> loc_sum(3);
    for (int i = 0; i < loc_sum.size(); i++)
        loc_sum[i] = (comm.rank() * loc_sum.size()) + i;
    std::vector<int> glob_sum;
    comm.reduce(loc_sum, glob_sum, op::sum<int>(), 0);

    if (comm.rank() == 0) {
        EXPECT_EQ(glob_sum.size(), 3);
        for (int i = 0; i < 3; i++) {
            int gold = n * (2 * i + (3 * (n - 1))) / 2;
            EXPECT_EQ(glob_sum[i], gold);
        }
    }
}

TEST(MPITest, reduce_all_sum)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int n = comm.size();
    int loc_sum = (comm.rank() + 1) * 3;
    int glob_sum;
    comm.all_reduce(loc_sum, glob_sum, op::sum<int>());
    comm.all_reduce(loc_sum, op::sum<int>());

    int gold = 3 * (n * (1 + n) / 2);
    EXPECT_EQ(loc_sum, gold);
    EXPECT_EQ(glob_sum, gold);
}

TEST(MPITest, reduce_all_prod)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int loc_prod = (comm.rank() + 1) * 3;
    int glob_prod;
    comm.all_reduce(loc_prod, glob_prod, op::prod<int>());
    comm.all_reduce(loc_prod, op::prod<int>());

    int gold = 1;
    for (int i = 1; i <= comm.size(); i++)
        gold *= 3 * i;

    EXPECT_EQ(loc_prod, gold);
    EXPECT_EQ(glob_prod, gold);
}

TEST(MPITest, reduce_all_min)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int loc = (comm.rank() + 1) * 3;
    int glob;
    comm.all_reduce(loc, glob, op::min<int>());
    comm.all_reduce(loc, op::min<int>());

    int gold = 3;
    EXPECT_EQ(loc, gold);
    EXPECT_EQ(glob, gold);
}

TEST(MPITest, reduce_all_max)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int loc = (comm.rank() + 1) * 3;
    int glob;
    comm.all_reduce(loc, glob, op::max<int>());
    comm.all_reduce(loc, op::max<int>());

    int gold = comm.size() * 3;
    EXPECT_EQ(loc, gold);
    EXPECT_EQ(glob, gold);
}

TEST(MPITest, reduce_all_logical_and)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    bool loc = false;
    bool glob;

    comm.all_reduce(loc, glob, op::logical_and<bool>());
    comm.all_reduce(loc, op::logical_and<bool>());
    EXPECT_FALSE(loc);
    EXPECT_FALSE(glob);

    loc = true;
    comm.all_reduce(loc, glob, op::logical_and<bool>());
    comm.all_reduce(loc, op::logical_and<bool>());
    EXPECT_TRUE(loc);
    EXPECT_TRUE(glob);

    if (comm.rank() == 0)
        loc = false;
    comm.all_reduce(loc, glob, op::logical_and<bool>());
    comm.all_reduce(loc, op::logical_and<bool>());
    EXPECT_FALSE(loc);
    EXPECT_FALSE(glob);
}

TEST(MPITest, reduce_all_logical_or)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    bool loc = false;
    bool glob;

    comm.all_reduce(loc, glob, op::logical_or<bool>());
    comm.all_reduce(loc, op::logical_or<bool>());
    EXPECT_FALSE(loc);
    EXPECT_FALSE(glob);

    loc = true;
    comm.all_reduce(loc, glob, op::logical_or<bool>());
    comm.all_reduce(loc, op::logical_or<bool>());
    EXPECT_TRUE(loc);
    EXPECT_TRUE(glob);

    if (comm.rank() != 0)
        loc = false;
    comm.all_reduce(loc, glob, op::logical_or<bool>());
    comm.all_reduce(loc, op::logical_or<bool>());
    EXPECT_TRUE(loc);
    EXPECT_TRUE(glob);
}

TEST(MPITest, reduce_all_logical_xor)
{
    Communicator comm;

    bool loc = false;
    bool glob;

    comm.all_reduce(loc, glob, op::logical_xor<bool>());
    comm.all_reduce(loc, op::logical_xor<bool>());
    EXPECT_FALSE(loc);
    EXPECT_FALSE(glob);

    loc = true;
    comm.all_reduce(loc, glob, op::logical_xor<bool>());
    comm.all_reduce(loc, op::logical_xor<bool>());
    if (comm.size() % 2 == 0) {
        EXPECT_FALSE(loc);
        EXPECT_FALSE(glob);
    }
    else {
        EXPECT_TRUE(loc);
        EXPECT_TRUE(glob);
    }
}

TEST(MPITest, iprobe)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        for (int i = 1; i < comm.size(); i++) {
            int num = i * 5;
            comm.send(i, tag, num);
        }
    }
    else {
        int timeout = 3;
        while (timeout > 0) {
            if (comm.iprobe(0, tag)) {
                int val;
                comm.recv(0, tag, val);
                EXPECT_EQ(val, comm.rank() * 5);
                SUCCEED();
                return;
            }
            else
                usleep(10000);
        }
        FAIL();
    }
}

TEST(MPITest, iprobe_w_status)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        for (int i = 1; i < comm.size(); i++) {
            int num = i * 5;
            comm.send(i, tag, num);
        }
    }
    else {
        int timeout = 3;
        while (timeout > 0) {
            Status status;
            if (comm.iprobe(0, tag, status)) {
                EXPECT_EQ(status.source(), 0);
                int val;
                comm.recv(0, tag, val);
                EXPECT_EQ(val, comm.rank() * 5);
                return;
            }
            else
                usleep(10000);
        }
        FAIL();
    }
}

TEST(MPITest, isend_irecv_wait)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        for (int i = 1; i < comm.size(); i++) {
            int num = i * 5;
            auto req = comm.isend(i, tag, num);
            wait(req);
        }
    }
    else {
        int val;
        auto req = comm.irecv(0, tag, val);
        wait(req);
        EXPECT_EQ(val, comm.rank() * 5);
    }
}

TEST(MPITest, isend_irecv_wait_w_status)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        for (int i = 1; i < comm.size(); i++) {
            int num = i * 5;
            auto request = comm.isend(i, tag, num);
            Status status;
            wait(request, status);
        }
    }
    else {
        int val;
        auto request = comm.irecv(0, tag, val);
        Status status;
        wait(request, status);
        EXPECT_EQ(val, comm.rank() * 5);
        EXPECT_EQ(status.source(), 0);
        EXPECT_EQ(status.tag(), tag);
    }
}

TEST(MPITest, isend_irecv_waitall)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        int n = comm.size() - 1;
        std::vector<int> vals;
        vals.resize(n);
        std::vector<Request> reqs;
        reqs.resize(n);
        for (int i = 0; i < n; i++)
            reqs[i] = comm.irecv(i + 1, tag, vals[i]);
        wait_all(reqs);
        for (int i = 0; i < n; i++)
            EXPECT_EQ(vals[i], (i + 1) * 7);
    }
    else {
        int num = comm.rank() * 7;
        auto req = comm.isend(0, tag, num);
        wait(req);
    }
}

TEST(MPITest, isend_irecv_waitany)
{
    Communicator comm;
    if (comm.size() == 1)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        int n = comm.size() - 1;
        std::vector<int> vals;
        vals.resize(n);
        std::vector<Request> reqs;
        reqs.resize(n);
        for (int i = 0; i < n; i++)
            reqs[i] = comm.irecv(i + 1, tag, vals[i]);

        auto idx = wait_any(reqs);
        EXPECT_EQ(vals[idx], (idx + 1) * 7);

        for (int i = 0; i < n; i++)
            if (i != idx)
                wait(reqs[i]);
    }
    else {
        int num = comm.rank() * 7;
        auto req = comm.isend(0, tag, num);
        wait(req);
    }
}

TEST(MPITest, test_all)
{
    Communicator comm;
    if (comm.size() < 2)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        int n = comm.size() - 1;
        std::vector<int> vals(n);
        std::vector<Request> reqs(n);
        for (int i = 0; i < n; i++)
            reqs[i] = comm.irecv(i + 1, tag, vals[i]);

        while (!test_all(reqs))
            ;
    }
    else {
        int num = comm.rank() * 7;
        auto req = comm.isend(0, tag, num);
        while (!test(req))
            ;
    }
}

TEST(MPITest, test_any)
{
    Communicator comm;
    if (comm.size() < 2)
        return;

    int tag = 1;
    if (comm.rank() == 0) {
        int n = comm.size() - 1;
        std::vector<int> vals(n);
        std::vector<Request> reqs(n);
        for (int i = 0; i < n; i++)
            reqs[i] = comm.irecv(i + 1, tag, vals[i]);

        int timeout = 2;
        while (timeout > 0) {
            std::size_t index = 0;
            if (test_any(reqs, index)) {
                SUCCEED();
                return;
            }
            else {
                usleep(10000);
                timeout--;
            }
        }
        FAIL();
    }
    else {
        int num = comm.rank() * 7;
        auto req = comm.isend(0, tag, num);
        while (!test(req))
            ;
    }
}

TEST(MPITest, all_to_all_1)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    std::vector<int> in_vals(comm.size());
    for (int i = 0; i < in_vals.size(); i++)
        in_vals[i] = (10 * comm.rank()) + i;
    std::vector<int> out_vals;
    comm.all_to_all(in_vals, out_vals);

    for (int i = 0; i < out_vals.size(); i++)
        EXPECT_EQ(out_vals[i], (10 * i) + comm.rank());
}

TEST(MPITest, all_to_all_vec)
{
    Communicator comm;
    if (comm.size() != 4)
        return;

    std::vector<std::vector<int>> in_vals(comm.size());
    if (comm.rank() == 0)
        in_vals = { { 1, 2 }, { 2 }, { -10, -11 }, {} };
    else if (comm.rank() == 1)
        in_vals = { { 3 }, { 4 }, { -12, -13, -14 }, { 100 } };
    else if (comm.rank() == 2)
        in_vals = { { 4, 5, 6 }, { 6 }, { -15 }, {} };
    else if (comm.rank() == 3)
        in_vals = { { 7, 8, 9, 10 }, { 8 }, { -16, -17 }, { 200 } };

    std::vector<int> out;
    comm.all_to_all(in_vals, out);

    if (comm.rank() == 0)
        EXPECT_THAT(out, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    else if (comm.rank() == 1)
        EXPECT_THAT(out, ElementsAre(2, 4, 6, 8));
    else if (comm.rank() == 2)
        EXPECT_THAT(out, ElementsAre(-10, -11, -12, -13, -14, -15, -16, -17));
    else if (comm.rank() == 3)
        EXPECT_THAT(out, ElementsAre(100, 200));
}

//

TEST(MPITest, status)
{
    Status s;
    EXPECT_EQ(s.error(), 0);
    EXPECT_EQ(s.tag(), 0);
    EXPECT_EQ(s.source(), 0);
}
