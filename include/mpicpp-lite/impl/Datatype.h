// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include "Environment.h"
#include <stdexcept>
#include <vector>
#include <cassert>

namespace mpicpp_lite {

/// Datatype traits for registering new types
template <typename T>
struct DatatypeTraits {
    /// Create a new datatype for MPI communication
    ///
    /// @tparam T Datatype
    /// @return New `MPI_Datatype`
    static MPI_Datatype
    get()
    {
        return MPI_DATATYPE_NULL;
    }
};

/// Register a new datatype for MPI communication
///
/// @tparam T Datatype to register
/// @return New MPI datatype
template <typename T>
MPI_Datatype
register_mpi_datatype()
{
    auto datatype = DatatypeTraits<T>::get();
    if (datatype == MPI_DATATYPE_NULL)
        throw std::runtime_error("Unknown type used in MPI communication");
    MPI_CHECK(MPI_Type_commit(&datatype));
    Environment::user_datatypes.push_back(datatype);
    return datatype;
}

/// General template to obtain an MPI_Datatype from a C++ type
///
/// @tparam T C++ data type
/// @return `MPI_Datatype` that is used in the MPI API
template <typename T>
inline MPI_Datatype
mpi_datatype()
{
    static auto dt = register_mpi_datatype<T>();
    return dt;
}

template <>
inline MPI_Datatype
mpi_datatype<char>()
{
    return MPI_BYTE;
}

template <>
inline MPI_Datatype
mpi_datatype<short>()
{
    return MPI_SHORT;
}

template <>
inline MPI_Datatype
mpi_datatype<int>()
{
    return MPI_INT;
}

template <>
inline MPI_Datatype
mpi_datatype<long int>()
{
    return MPI_LONG;
}

template <>
inline MPI_Datatype
mpi_datatype<long long int>()
{
    return MPI_LONG_LONG;
}

template <>
inline MPI_Datatype
mpi_datatype<unsigned char>()
{
    return MPI_UNSIGNED_CHAR;
}

template <>
inline MPI_Datatype
mpi_datatype<unsigned short>()
{
    return MPI_UNSIGNED_SHORT;
}

template <>
inline MPI_Datatype
mpi_datatype<unsigned int>()
{
    return MPI_UNSIGNED;
}

template <>
inline MPI_Datatype
mpi_datatype<unsigned long int>()
{
    return MPI_UNSIGNED_LONG;
}

template <>
inline MPI_Datatype
mpi_datatype<unsigned long long int>()
{
    return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline MPI_Datatype
mpi_datatype<float>()
{
    return MPI_FLOAT;
}

template <>
inline MPI_Datatype
mpi_datatype<double>()
{
    return MPI_DOUBLE;
}

template <>
inline MPI_Datatype
mpi_datatype<long double>()
{
    return MPI_LONG_DOUBLE;
}

template <>
inline MPI_Datatype
mpi_datatype<bool>()
{
    return MPI_CXX_BOOL;
}

#if __cplusplus >= 201703L

template <>
inline MPI_Datatype
mpi_datatype<std::byte>()
{
    return MPI_BYTE;
}

#endif

template <typename T>
[[deprecated("use `mpi_datatype` instead")]] inline MPI_Datatype
get_mpi_datatype()
{
    return mpi_datatype<T>();
}

/// Create a new datatype for MPI communication
///
/// @param types MPI datatypes
/// @param blk_lens Number of elements of each type
/// @param offsets Byte offsets of each element
/// @return New MPI datatype
inline MPI_Datatype
type_create_struct(const std::vector<MPI_Datatype> & types,
                   const std::vector<int> & blk_lens,
                   const std::vector<MPI_Aint> & offsets)
{
    assert(types.size() == blk_lens.size());
    assert(types.size() == offsets.size());
    MPI_Datatype dt;
    MPI_CHECK(
        MPI_Type_create_struct(types.size(), blk_lens.data(), offsets.data(), types.data(), &dt));
    return dt;
}

/// Creates a contiguous datatype
///
/// @param count Replication count (nonnegative integer)
/// @param type Datatype (handle)
/// @return New MPI datatype
inline MPI_Datatype
type_contiguous(int count, MPI_Datatype type)
{
    MPI_Datatype dt;
    MPI_CHECK(MPI_Type_contiguous(count, type, &dt));
    return dt;
}

/// Return the number of bytes occupied by entries in the datatype
///
/// @tparam T Datatype
/// @return Number of bytes
template <typename T>
inline int
type_size()
{
    auto dt = mpi_datatype<T>();
    int size;
    MPI_CHECK(MPI_Type_size(dt, &size));
    return size;
}

} // namespace mpicpp_lite
