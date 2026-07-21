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

/// Wait for a single request to complete
///
/// @param request Request to wait for
/// @return Status of the operation
inline Status
wait(Request & request)
{
    Status status;
    MPI_CHECK(MPI_Wait(&request.native(), &status.native()));
    return status;
}

/// Wait for a single request to complete with status
///
/// @param request Request to wait for
/// @param status Status
inline void
wait(Request & request, Status & status)
{
    MPI_CHECK(MPI_Wait(&request.native(), &status.native()));
}

/// Wait for a single request with a timeout
///
/// @param request MPI request to wait on
/// @param timeout Timeout in seconds
/// @return `true` if request completed within given timeout, `false` otherwise
inline bool
wait_with_timeout(Request & request, double timeout)
{
    bool completed = false;
    auto start = wall_time();
    while (!completed && ((wall_time() - start) < timeout)) {
        // Busy wait
        completed = test(request).has_value();
    }
    return completed;
}

/// Wait for all requests to complete
///
/// @param requests Requests to wait for
inline void
wait_all(std::vector<Request> & requests)
{
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    MPI_CHECK(MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE));
}

/// Wait for any specified request to complete
///
/// @param requests Requests to wait for
/// @return Index of the request that completed, or `UNDEFINED` if all requests are null requests
inline TestAnyResult
wait_any(std::vector<Request> & requests)
{
    TestAnyResult result;
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    MPI_CHECK(MPI_Waitany(n, reqs, &result.index, &result.status.native()));
    return result;
}

/// Waits for some given requests to complete
///
/// @param requests Array of requests
/// @param indices Array of indices of operations that completed
/// @return `false` If there is no active handle in the list, otherwise `true`
///         (in this case the length of indices indicates how many operations
///         completed)
inline bool
wait_some(std::vector<Request> & requests, std::vector<int> & indices)
{
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    indices.resize(requests.size());
    int outcount = UNDEFINED;
    MPI_CHECK(MPI_Waitsome(static_cast<int>(requests.size()),
                           reqs,
                           &outcount,
                           indices.data(),
                           MPI_STATUSES_IGNORE));
    if (outcount != UNDEFINED) {
        indices.resize(static_cast<std::size_t>(outcount));
        return true;
    }
    else
        return false;
}

/// Waits for some given requests to complete
///
/// @param requests Array of requests
/// @param indices Array of indices of operations that completed
/// @param statuses Array of status objects for operations that completed
/// @return `false` If there is no active handle in the list, otherwise `true`
///         (in this case the length of indices indicates how many operations
///         completed)
inline bool
wait_some(std::vector<Request> & requests,
          std::vector<int> & indices,
          std::vector<Status> & statuses)
{
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    indices.resize(requests.size());
    statuses.resize(requests.size());
    auto * stats = reinterpret_cast<MPI_Status *>(statuses.data());
    int outcount = UNDEFINED;
    MPI_CHECK(
        MPI_Waitsome(static_cast<int>(requests.size()), reqs, &outcount, indices.data(), stats));
    if (outcount != UNDEFINED) {
        auto cnt = static_cast<std::size_t>(outcount);
        indices.resize(cnt);
        statuses.resize(cnt);
        return true;
    }
    else
        return false;
}

} // namespace mpicpp_lite
