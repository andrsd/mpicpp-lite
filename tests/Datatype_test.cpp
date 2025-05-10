#include "gmock/gmock.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <vector>
#include <array>

namespace mpi = mpicpp_lite;

TEST(MPITest, std_datatypes)
{
    EXPECT_EQ(mpi::mpi_datatype<char>(), MPI_BYTE);
    EXPECT_EQ(mpi::mpi_datatype<short>(), MPI_SHORT);
    EXPECT_EQ(mpi::mpi_datatype<int>(), MPI_INT);
    EXPECT_EQ(mpi::mpi_datatype<long int>(), MPI_LONG);
    EXPECT_EQ(mpi::mpi_datatype<long long int>(), MPI_LONG_LONG);
    EXPECT_EQ(mpi::mpi_datatype<unsigned char>(), MPI_UNSIGNED_CHAR);
    EXPECT_EQ(mpi::mpi_datatype<unsigned short>(), MPI_UNSIGNED_SHORT);
    EXPECT_EQ(mpi::mpi_datatype<unsigned int>(), MPI_UNSIGNED);
    EXPECT_EQ(mpi::mpi_datatype<unsigned long int>(), MPI_UNSIGNED_LONG);
    EXPECT_EQ(mpi::mpi_datatype<unsigned long long int>(), MPI_UNSIGNED_LONG_LONG);
    EXPECT_EQ(mpi::mpi_datatype<float>(), MPI_FLOAT);
    EXPECT_EQ(mpi::mpi_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(mpi::mpi_datatype<long double>(), MPI_LONG_DOUBLE);
}

namespace {

// CustomData structure

struct CustomData {
    int id;
    double value;
    bool b;
    char name[20];
};

// Custom enum

enum class CustomEnum : unsigned char { RED, BLUE, GREEN };

// Templated type

template <typename T, int D>
struct DVec {
    std::array<T, D> vals;
};

} // namespace

namespace mpicpp_lite {

template <>
struct DatatypeTraits<CustomData> {
    static MPI_Datatype
    get()
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
};

template <>
struct DatatypeTraits<CustomEnum> {
    static MPI_Datatype
    get()
    {
        return type_contiguous(1, mpi_datatype<unsigned char>());
    }
};

template <typename T, int D>
struct DatatypeTraits<DVec<T, D>> {
    static MPI_Datatype
    get()
    {
        return type_contiguous(D, mpi::mpi_datatype<T>());
    }
};

namespace op {

template <>
struct sum<CustomData> : public UserOp<sum<CustomData>, CustomData> {
public:
    CustomData
    operator()(const CustomData & x, const CustomData & y) const
    {
        return { 0, x.value + y.value, true, "empty" };
    }
};

template <>
struct IsCommutative<logical_and<CustomEnum>, CustomEnum> : public std::true_type {};

template <>
struct logical_and<CustomEnum> : public UserOp<logical_and<CustomEnum>, CustomEnum> {
public:
    CustomEnum
    operator()(const CustomEnum & x, const CustomEnum & y) const
    {
        return CustomEnum::RED;
    }
};

template <>
struct logical_or<CustomEnum> : public UserOp<logical_or<CustomEnum>, CustomEnum> {
public:
    CustomEnum
    operator()(const CustomEnum & x, const CustomEnum & y) const
    {
        return CustomEnum::BLUE;
    }
};

} // namespace op

} // namespace mpicpp_lite

TEST(DatatypeTest, custom_struct)
{
    mpi::Communicator comm;
    if (comm.size() != 4)
        return;

    CustomData data;
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

TEST(DatatypeTest, custom_op_reduce)
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

    CustomEnum f = CustomEnum::GREEN;
    comm.reduce(e, f, mpi::op::logical_and<CustomEnum>(), 0);
    if (comm.rank() == 0)
        EXPECT_EQ(f, CustomEnum::RED);
}

TEST(DatatypeTest, custom_op_all_reduce)
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

    CustomEnum f = CustomEnum::GREEN;
    comm.all_reduce(e, f, mpi::op::logical_or<CustomEnum>());
    EXPECT_EQ(f, CustomEnum::BLUE);
}

TEST(DatatypeTest, custom_op_scan)
{
    mpi::Communicator comm;
    if (comm.size() != 4)
        return;

    CustomData d = { comm.rank(), (comm.rank() + 1) * 2., false, "text" };
    CustomData e = { 1234, 5678., true, "A" };
    comm.scan(d, e, mpi::op::sum<CustomData>());
    if (comm.rank() == 0) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 2., 1e-10);
        EXPECT_FALSE(e.b);
        EXPECT_STREQ(e.name, "text");
    }
    else if (comm.rank() == 1) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 6., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "empty");
    }
    else if (comm.rank() == 2) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 12., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "empty");
    }
    else if (comm.rank() == 3) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 20., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "empty");
    }
}

TEST(DatatypeTest, custom_op_exscan)
{
    mpi::Communicator comm;
    if (comm.size() != 4)
        return;

    CustomData d = { comm.rank(), (comm.rank() + 1) * 2., false, "text" };
    CustomData e = { 4321, 8765., true, "B" };
    comm.exscan(d, e, mpi::op::sum<CustomData>());

    if (comm.rank() == 0) {
        EXPECT_EQ(e.id, 4321);
        EXPECT_NEAR(e.value, 8765., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "B");
    }
    else if (comm.rank() == 1) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 2., 1e-10);
        EXPECT_FALSE(e.b);
        EXPECT_STREQ(e.name, "text");
    }
    else if (comm.rank() == 2) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 6., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "empty");
    }
    else if (comm.rank() == 3) {
        EXPECT_EQ(e.id, 0);
        EXPECT_NEAR(e.value, 12., 1e-10);
        EXPECT_TRUE(e.b);
        EXPECT_STREQ(e.name, "empty");
    }
}

TEST(DatatypeTest, type_size)
{
    mpi::Communicator comm;
    EXPECT_EQ(mpi::type_size<long>(), 8);
    EXPECT_EQ(mpi::type_size<CustomData>(), 33);
    EXPECT_EQ(mpi::type_size<CustomEnum>(), 1);
}

TEST(DatatypeTest, custom_templated_type)
{
    mpi::Communicator comm;
    if (comm.size() > 1)
        return;

    DVec<int, 2> data;
    if (comm.rank() == 0) {
        data.vals[0] = 42;
        data.vals[1] = 12;
    }
    comm.broadcast(data, 0);

    EXPECT_EQ(data.vals[0], 42);
    EXPECT_EQ(data.vals[1], 12);
}
