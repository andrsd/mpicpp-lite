// SPDX-FileCopyrightText: 2025 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"

namespace mpicpp_lite {

/// Returns an elapsed time on the calling processor
///
/// @return Time in seconds since an arbitrary time in the past.
inline double
wall_time()
{
    return MPI_Wtime();
}

} // namespace mpicpp_lite
