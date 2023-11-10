#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"

using namespace mpicpp_lite;


TEST(MPITest, std_datatypes)
{
    EXPECT_EQ(get_mpi_datatype<char>(), MPI_BYTE);
    EXPECT_EQ(get_mpi_datatype<short>(), MPI_SHORT);
    EXPECT_EQ(get_mpi_datatype<int>(), MPI_INT);
    EXPECT_EQ(get_mpi_datatype<long int>(), MPI_LONG);
    EXPECT_EQ(get_mpi_datatype<long long int>(), MPI_LONG_LONG);
    EXPECT_EQ(get_mpi_datatype<unsigned char>(), MPI_UNSIGNED_CHAR);
    EXPECT_EQ(get_mpi_datatype<unsigned short>(), MPI_UNSIGNED_SHORT);
    EXPECT_EQ(get_mpi_datatype<unsigned int>(), MPI_UNSIGNED);
    EXPECT_EQ(get_mpi_datatype<unsigned long int>(), MPI_UNSIGNED_LONG);
    EXPECT_EQ(get_mpi_datatype<unsigned long long int>(), MPI_UNSIGNED_LONG_LONG);
    EXPECT_EQ(get_mpi_datatype<float>(), MPI_FLOAT);
    EXPECT_EQ(get_mpi_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(get_mpi_datatype<long double>(), MPI_LONG_DOUBLE);
}
