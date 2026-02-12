// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include <type_traits>
#include <vector>
#include "Request.h"
#include "Status.h"
#include "Error.h"
#include "Time.h"
#include "Test.h"

namespace mpicpp_lite {

/// Wait for a single request to complete, ignoring status
///
/// @param request Request to wait for
inline void
wait(const Request & request)
{
    MPI_CHECK(MPI_Wait(const_cast<Request &>(request), MPI_STATUS_IGNORE));
}

/// Wait for a single request to complete with status
///
/// @param request Request to wait for
/// @param status Status
inline void
wait(const Request & request, Status & status)
{
    MPI_CHECK(MPI_Wait(const_cast<Request &>(request), status));
}

/// Wait for a single request with a timeout
///
/// @param request MPI request to wait on
/// @param timeout Timeout in seconds
/// @return `true` if request completed within given timeout, `false` otherwise
inline bool
wait_with_timeout(const Request & request, double timeout)
{
    bool completed = false;
    auto start = wall_time();
    while (!completed && ((wall_time() - start) < timeout)) {
        // Busy wait
        completed = test(request);
    }
    return completed;
}

/// Wait for all requests to complete
///
/// @param requests Requests to wait for
inline void
wait_all(const std::vector<Request> & requests)
{
    static_assert(sizeof(Request) == sizeof(MPI_Request),
                  "Size of `Request` must match `MPI_request`");
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(const_cast<Request *>(requests.data()));
    MPI_CHECK(MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE));
}

/// Wait for any specified request to complete
///
/// @param requests Requests to wait for
/// @return Index of the request that completed
inline std::size_t
wait_any(const std::vector<Request> & requests)
{
    static_assert(sizeof(Request) == sizeof(MPI_Request),
                  "Size of `Request` must match `MPI_request`");
    auto n = static_cast<int>(requests.size());
    int idx;
    auto * reqs = reinterpret_cast<MPI_Request *>(const_cast<Request *>(requests.data()));
    MPI_CHECK(MPI_Waitany(n, reqs, &idx, MPI_STATUS_IGNORE));
    return idx;
}

} // namespace mpicpp_lite
