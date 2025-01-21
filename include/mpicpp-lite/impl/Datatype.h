// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"

namespace mpicpp_lite {

/// Create a new datatype for MPI communication
///
/// @tparam T Datatype
/// @return New `MPI_Datatype`
template <typename T>
inline MPI_Datatype
create_mpi_datatype()
{
    return MPI_DATATYPE_NULL;
}

/// General template to obtain an MPI_Datatype from a C++ type
///
/// @tparam T C++ data type
/// @return `MPI_Datatype` that is used in the MPI API
template <typename T>
inline MPI_Datatype
get_mpi_datatype()
{
    return MPI_DATATYPE_NULL;
}

template <>
inline MPI_Datatype
get_mpi_datatype<char>()
{
    return MPI_BYTE;
}

template <>
inline MPI_Datatype
get_mpi_datatype<short>()
{
    return MPI_SHORT;
}

template <>
inline MPI_Datatype
get_mpi_datatype<int>()
{
    return MPI_INT;
}

template <>
inline MPI_Datatype
get_mpi_datatype<long int>()
{
    return MPI_LONG;
}

template <>
inline MPI_Datatype
get_mpi_datatype<long long int>()
{
    return MPI_LONG_LONG;
}

template <>
inline MPI_Datatype
get_mpi_datatype<unsigned char>()
{
    return MPI_UNSIGNED_CHAR;
}

template <>
inline MPI_Datatype
get_mpi_datatype<unsigned short>()
{
    return MPI_UNSIGNED_SHORT;
}

template <>
inline MPI_Datatype
get_mpi_datatype<unsigned int>()
{
    return MPI_UNSIGNED;
}

template <>
inline MPI_Datatype
get_mpi_datatype<unsigned long int>()
{
    return MPI_UNSIGNED_LONG;
}

template <>
inline MPI_Datatype
get_mpi_datatype<unsigned long long int>()
{
    return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline MPI_Datatype
get_mpi_datatype<float>()
{
    return MPI_FLOAT;
}

template <>
inline MPI_Datatype
get_mpi_datatype<double>()
{
    return MPI_DOUBLE;
}

template <>
inline MPI_Datatype
get_mpi_datatype<long double>()
{
    return MPI_LONG_DOUBLE;
}

template <>
inline MPI_Datatype
get_mpi_datatype<bool>()
{
    return MPI_CXX_BOOL;
}

#if __cplusplus >= 201703L

template <>
inline MPI_Datatype
get_mpi_datatype<std::byte>()
{
    return MPI_BYTE;
}

#endif

/// Register a new datatype for MPI communication
///
/// @tparam T Datatype to register
/// @return New MPI datatype
template <typename T>
MPI_Datatype
register_mpi_datatype()
{
    auto datatype = create_mpi_datatype<T>();
    MPI_CHECK(MPI_Type_commit(&datatype));
    return datatype;
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
    auto dt = get_mpi_datatype<T>();
    int size;
    MPI_CHECK(MPI_Type_size(dt, &size));
    return size;
}

} // namespace mpicpp_lite
