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

    /// Type cast operator so we can pass this class directly into MPI calls
    operator MPI_Request() const { return this->request; }

private:
    MPI_Request request;

    friend class Communicator;
    friend void wait(Request &);
    friend void wait(Request &, Status &);
    friend bool test(Request &);
    friend bool test(Request &, Status &);
};

inline Request::Request() : request(MPI_REQUEST_NULL)
{
    static_assert(sizeof(Request) == sizeof(MPI_Request),
                  "Size of `Request` must match `MPI_request`");
}

inline void
Request::cancel()
{
    MPI_CHECK(MPI_Cancel(&this->request));
}

} // namespace mpicpp_lite
