#pragma once

#include "fmt/printf.h"
#include "mpi.h"

namespace mpicpp_lite {

namespace internal {

/// Terminate the run
inline void
terminate(MPI_Comm comm, int status = 1)
{
    MPI_Abort(comm, status);
}

inline void
check_mpi_error(MPI_Comm comm, int ierr, const char * file, int line)
{
    if (ierr) {
        fmt::print(stderr, "[ERROR] MPI error {} at {}:{}", ierr, file, line);
        terminate(comm, ierr);
    }
}

} // namespace internal

/// Check that MPI call was successful.
#define MPI_CHECK_SELF(ierr) mpicpp_lite::internal::check_mpi_error(this->comm, ierr, __FILE__, __LINE__)

#define MPI_CHECK(ierr) mpicpp_lite::internal::check_mpi_error(MPI_COMM_WORLD, ierr, __FILE__, __LINE__)

} // namespace mpicpp_lite
