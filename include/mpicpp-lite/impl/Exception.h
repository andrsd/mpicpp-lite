// SPDX-FileCopyrightText: 2025 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "Communicator.h"
#include "Wait.h"
#include "mpi.h"
#include <exception>
#include <string>

namespace mpicpp_lite {

class ParallelException : public std::exception {
public:
    ParallelException() = default;
    ParallelException(int rank) : rank_(rank) {}
    ParallelException(int rank, const std::string & msg) : rank_(rank), msg_(msg) {}

    /// Rank that this exception was thrown at
    int
    rank() const
    {
        return this->rank_;
    }

    /// Error message for this exception
    const char *
    what() const noexcept override
    {
        return this->msg_.c_str();
    }

    /// Broadcast the exception accross the communicator
    ///
    /// @param comm Communicator
    void
    broadcast(const Communicator & comm)
    {
        comm.broadcast(this->msg_, this->rank_);
    }

private:
    int rank_ = -1;
    std::string msg_;
};

/// Handle exceptions accross an MPI communicator
///
/// If a rank needs to throw an exception, in normal execution flow, the code gets out
/// of sync. This class helps with keeping the code in sync. Users can use standard
/// try-catch block with this class and simply keep the execution in sync.
///
/// Usage:
/// ```
/// CollectiveExceptionHandler handler(comm);
/// try {
///     // random rank throws
///     if (comm.rank() == 1) {
///         throw std::runtime_error("my error");
///     }
///
///     handler.sync();
/// }
/// catch (std::exception & e) {
///     auto par_exc = handler.sync(e);
///     // from now on, we are in sync again
///
/// }
/// ```
///
/// This is based on: Jaros, Jiri. "Handling C++ Exceptions in MPI Applications." [1]
/// [1] https://github.com/jarosjir/MPIErrorChecker
///
class CollectiveExceptionHandler {
public:
    CollectiveExceptionHandler(Communicator comm) : comm_(comm), error_xch_comm_(comm.duplicate())
    {
    }

    ~CollectiveExceptionHandler() { this->error_xch_comm_.free(); }

    /// Sync point (use inside the catch-block)
    ///
    /// @param e Exception that was caught
    const ParallelException
    sync(const std::exception & e)
    {
        if (typeid(e) == typeid(ParallelException)) {
            // we get here when `finalize()` threw
            return dynamic_cast<const ParallelException &>(e);
        }
        else {
            int rank = this->error_xch_comm_.rank();
            int faulty_rank;
            auto req = this->error_xch_comm_.iall_reduce(rank, faulty_rank, op::max<int>());
            auto completed = wait_with_timeout(req, DEFAULT_TIMEOUT);

            if (completed) {
                if (rank == faulty_rank)
                    // exception was thrown on our rank
                    return ParallelException(faulty_rank, e.what());
                else
                    // exception was thrown on our rank, but another rank also threw
                    return ParallelException(faulty_rank);
            }
            else
                // some rank threw, but we don't know who (timeout happened)
                // TODO: indicate a hang
                return ParallelException();
        }
    }

    /// Sync point (use inside the try-block)
    void
    sync()
    {
        int rank = INVALID_RANK;
        int faulty_rank = INVALID_RANK;
        auto req = this->error_xch_comm_.iall_reduce(rank, faulty_rank, op::max<int>());
        wait(req);

        if (faulty_rank != INVALID_RANK)
            // some rank threw, but not us -> throw a parallel exception
            throw ParallelException(faulty_rank);
    }

private:
    /// Main communicator
    Communicator comm_;
    /// Communicator for exchanging errors (duplicate of the main comm)
    Communicator error_xch_comm_;

private:
    static inline constexpr int INVALID_RANK = -1;
    /// Timeout in seconds
    static inline constexpr double DEFAULT_TIMEOUT = 5.;
};

} // namespace mpicpp_lite
