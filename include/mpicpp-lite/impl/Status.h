// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Datatype.h"
#include "Tag.h"

namespace mpicpp_lite {

class Request;

/// Wrapper around MPI_Status
class Status {
public:
    /// Construct empty `Status` object
    Status();

    /// Get the source of the message
    ///
    /// @return Source of the message (i.e. rank ID)
    int source() const;

    /// Get the message tag
    ///
    /// @return Message tag
    Tag tag() const;

    /// Get the error code
    ///
    /// @return Error code
    int error() const;

    template <typename T>
    [[deprecated("use `count` instead")]] int
    get_count() const
    {
        return count<T>();
    }

    /// Gets the number of "top level" elements
    ///
    /// @tparam T datatype of each receive buffer element
    /// @return The number of "top level" elements
    template <typename T>
    int count() const;

    MPI_Status &
    native()
    {
        return this->status_;
    }

    const MPI_Status &
    native() const
    {
        return this->status_;
    }

private:
    MPI_Status status_;
};

inline Status::Status() : status_({ 0 }) {}

inline int
Status::source() const
{
    return this->status_.MPI_SOURCE;
}

inline Tag
Status::tag() const
{
    return Tag(this->status_.MPI_TAG);
}

inline int
Status::error() const
{
    return this->status_.MPI_ERROR;
}

template <typename T>
inline int
Status::count() const
{
    int n;
    MPI_Get_count(&this->status_, mpi_datatype<T>(), &n);
    return n;
}

} // namespace mpicpp_lite
