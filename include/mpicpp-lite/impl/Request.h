// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"

namespace mpicpp_lite {

class Status;

/// Wrapper around MPI_Request
class Request {
public:
    /// Create empty request
    Request();

    /// Cancels a communication request
    void cancel();

    MPI_Request &
    native()
    {
        return this->request_;
    }

    const MPI_Request &
    native() const
    {
        return this->request_;
    }

private:
    MPI_Request request_;
};

inline Request::Request() : request_(MPI_REQUEST_NULL)
{
    static_assert(sizeof(Request) == sizeof(MPI_Request),
                  "Size of `Request` must match `MPI_request`");
}

inline void
Request::cancel()
{
    MPI_CHECK(MPI_Cancel(&this->request_));
}

} // namespace mpicpp_lite
