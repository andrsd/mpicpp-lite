// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include <vector>
#include "Request.h"
#include "Status.h"
#include "Error.h"
#include "Enums.h"

namespace mpicpp_lite {

/// Test for the completion of a request
///
/// @param request Request to test
/// @return `true` if operation completed, `false` otherwise
inline bool
test(Request & request)
{
    int flag;
    MPI_CHECK(MPI_Test(request, &flag, MPI_STATUS_IGNORE));
    return flag != 0;
}

/// Test for the completion of a request with status
///
/// @param request Request to wait for
/// @param status Status
/// @return `true` if operation completed, `false` otherwise
inline bool
test(Request & request, Status & status)
{
    int flag;
    MPI_CHECK(MPI_Test(request, &flag, status));
    return flag != 0;
}

/// Test for the completion of all previously initiated requests
///
/// @param requests Requests to test
/// @return `true` only if all requests have completed, `false` otherwise
inline bool
test_all(std::vector<Request> & requests)
{
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    int flag;
    MPI_CHECK(MPI_Testall(n, reqs, &flag, MPI_STATUSES_IGNORE));
    return flag != 0;
}

/// Test for completion of any previously initiated requests
///
/// @param requests Requests to test
/// @param index Index of operation that completed or `UNDEFINED` if none completed
/// @return `true` if one of the operations is complete
inline bool
test_any(std::vector<Request> & requests, int & index)
{
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    int flag;
    MPI_CHECK(MPI_Testany(n, reqs, &index, &flag, MPI_STATUSES_IGNORE));
    return flag != 0;
}

/// Tests for some given requests to complete
///
/// @param requests Array of requests
/// @param indices Array of indices of operations that completed
/// @return `false` If there is no active handle in the list, otherwise `true`
///         (in this case the length of indices indicates how many operations
///         completed)
inline bool
test_some(std::vector<Request> & requests, std::vector<int> & indices)
{
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    indices.resize(requests.size());
    int outcount = UNDEFINED;
    MPI_CHECK(MPI_Testsome(static_cast<int>(requests.size()),
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

/// Tests for some given requests to complete
///
/// @param requests Array of requests
/// @param indices Array of indices of operations that completed
/// @param statuses Array of status objects for operations that completed
/// @return `false` If there is no active handle in the list, otherwise `true`
///         (in this case the length of indices indicates how many operations
///         completed)
inline bool
test_some(std::vector<Request> & requests,
          std::vector<int> & indices,
          std::vector<Status> & statuses)
{
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    indices.resize(requests.size());
    statuses.resize(requests.size());
    auto * stats = reinterpret_cast<MPI_Status *>(statuses.data());
    int outcount = UNDEFINED;
    MPI_CHECK(
        MPI_Testsome(static_cast<int>(requests.size()), reqs, &outcount, indices.data(), stats));
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
