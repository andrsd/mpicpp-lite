#pragma once

#include "mpi.h"

namespace mpicpp_lite {

/// Wrapper around MPI_Request
class Request {
public:
    /// Create empty request
    Request();
    /// Create request from an MPI_Request
    Request(const MPI_Request & r);

    /// Type cast operators so we can pass this class directly into MPI calls
    operator MPI_Request *() { return &this->request; }
    operator const MPI_Request &() const { return this->request; }

private:
    MPI_Request request;
};

inline Request::Request() {}

inline Request::Request(const MPI_Request & r) : request(r) {}

} // namespace mpi
