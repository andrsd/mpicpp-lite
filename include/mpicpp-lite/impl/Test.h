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
/// @return `Status` if operation completed, `nullopt` otherwise
inline std::optional<Status>
test(Request & request)
{
    int flag;
    Status status;
    MPI_CHECK(MPI_Test(&request.native(), &flag, &status.native()));
    if (flag != 0)
        return status;
    else
        return std::nullopt;
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
    MPI_CHECK(MPI_Test(&request.native(), &flag, &status.native()));
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

struct TestAnyResult {
    /// Index of operation that completed or `UNDEFINED` if none completed
    int index = UNDEFINED;
    /// Status of the operation
    Status status;
};

/// Test for completion of any previously initiated requests
///
/// @param requests Requests to test
/// @return `TestAnyResult` if one of the operations is complete
inline std::optional<TestAnyResult>
test_any(std::vector<Request> & requests)
{
    TestAnyResult result;
    auto n = static_cast<int>(requests.size());
    auto * reqs = reinterpret_cast<MPI_Request *>(requests.data());
    int flag;
    MPI_CHECK(MPI_Testany(n, reqs, &result.index, &flag, &result.status.native()));
    if (flag != 0)
        return result;
    else
        return std::nullopt;
}

} // namespace mpicpp_lite
