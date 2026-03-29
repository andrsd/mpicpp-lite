// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include <vector>
#include "Request.h"
#include "Status.h"
#include "Error.h"

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
    MPI_CHECK(MPI_Test(request, &flag, &status.status));
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

} // namespace mpicpp_lite
