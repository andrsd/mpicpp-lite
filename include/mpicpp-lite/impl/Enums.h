// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"

namespace mpicpp_lite {

enum { UNDEFINED = MPI_UNDEFINED, ANY_SOURCE = MPI_ANY_SOURCE, ANY_TAG = MPI_ANY_TAG };

enum class CommType {
    /// This type splits the communicator into subcommunicators, each of which can create a shared
    /// memory region
    SHARED = MPI_COMM_TYPE_SHARED,
#if MPI_VERSION >= 4
    ///
    HW_GUIDED = MPI_COMM_TYPE_HW_GUIDED,
    ///
    RESOURCE_GUIDED = MPI_COMM_TYPE_RESOURCE_GUIDED,
#endif
};

} // namespace mpicpp_lite
