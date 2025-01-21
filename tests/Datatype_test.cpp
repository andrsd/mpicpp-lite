#include "gmock/gmock.h"
#include "mpicpp-lite/impl/Datatype.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <vector>

namespace mpi = mpicpp_lite;

TEST(MPITest, std_datatypes)
{
    EXPECT_EQ(mpi::get_mpi_datatype<char>(), MPI_BYTE);
    EXPECT_EQ(mpi::get_mpi_datatype<short>(), MPI_SHORT);
    EXPECT_EQ(mpi::get_mpi_datatype<int>(), MPI_INT);
    EXPECT_EQ(mpi::get_mpi_datatype<long int>(), MPI_LONG);
    EXPECT_EQ(mpi::get_mpi_datatype<long long int>(), MPI_LONG_LONG);
    EXPECT_EQ(mpi::get_mpi_datatype<unsigned char>(), MPI_UNSIGNED_CHAR);
    EXPECT_EQ(mpi::get_mpi_datatype<unsigned short>(), MPI_UNSIGNED_SHORT);
    EXPECT_EQ(mpi::get_mpi_datatype<unsigned int>(), MPI_UNSIGNED);
    EXPECT_EQ(mpi::get_mpi_datatype<unsigned long int>(), MPI_UNSIGNED_LONG);
    EXPECT_EQ(mpi::get_mpi_datatype<unsigned long long int>(), MPI_UNSIGNED_LONG_LONG);
    EXPECT_EQ(mpi::get_mpi_datatype<float>(), MPI_FLOAT);
    EXPECT_EQ(mpi::get_mpi_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(mpi::get_mpi_datatype<long double>(), MPI_LONG_DOUBLE);
}

namespace {

struct CustomData {
    int id;
    double value;
    bool b;
    char name[20];
};

enum class CustomEnum : unsigned char { RED, BLUE, GREEN };

} // namespace

namespace mpicpp_lite {

template <>
inline MPI_Datatype
create_mpi_datatype<CustomData>()
{
    std::vector<MPI_Datatype> types = { MPI_INT, MPI_DOUBLE, MPI_CXX_BOOL, MPI_CHAR };
    std::vector<int> blk_lens = { 1, 1, 1, 20 };
    std::vector<MPI_Aint> offsets = { offsetof(CustomData, id),
                                      offsetof(CustomData, value),
                                      offsetof(CustomData, b),
                                      offsetof(CustomData, name) };
    auto dt = type_create_struct(types, blk_lens, offsets);
    return dt;
}

template <>
inline MPI_Datatype
get_mpi_datatype<CustomEnum>()
{
    static auto dt = mpi::register_mpi_datatype<CustomEnum>();
    return dt;
}

template <>
inline MPI_Datatype
create_mpi_datatype<CustomEnum>()
{
    return type_contiguous(1, get_mpi_datatype<unsigned char>());
}

template <>
inline MPI_Datatype
get_mpi_datatype<CustomData>()
{
    static auto dt = mpi::register_mpi_datatype<CustomData>();
    return dt;
}

} // namespace mpicpp_lite

TEST(DatatypeTest, custom_struct)
{
    mpi::Communicator comm;
    if (comm.size() != 4)
        return;

    CustomData data;
    int tag = 1234;
    if (comm.rank() == 0) {
        data.id = 42;
        data.value = 3.14;
        data.b = true;
        strncpy(data.name, "hello", 20);
    }
    comm.broadcast(data, 0);

    EXPECT_EQ(data.id, 42);
    EXPECT_NEAR(data.value, 3.14, 1e-10);
    EXPECT_TRUE(data.b);
    EXPECT_STREQ(data.name, "hello");
}

TEST(DatatypeTest, custom_enum)
{
    mpi::Communicator comm;
    if (comm.size() != 4)
        return;

    CustomEnum e;
    if (comm.rank() == 0)
        e = CustomEnum::GREEN;
    else if (comm.rank() == 1)
        e = CustomEnum::BLUE;
    else if (comm.rank() == 2)
        e = CustomEnum::RED;
    else
        e = CustomEnum::GREEN;

    std::vector<CustomEnum> colors;
    comm.gather(e, colors, 0);
    if (comm.rank() == 0) {
        EXPECT_THAT(colors,
                    testing::ElementsAre(CustomEnum::GREEN,
                                         CustomEnum::BLUE,
                                         CustomEnum::RED,
                                         CustomEnum::GREEN));
    }
}

TEST(DatatypeTest, type_size)
{
    mpi::Communicator comm;
    EXPECT_EQ(mpi::type_size<long>(), 8);
    EXPECT_EQ(mpi::type_size<CustomData>(), 33);
    EXPECT_EQ(mpi::type_size<CustomEnum>(), 1);
}
