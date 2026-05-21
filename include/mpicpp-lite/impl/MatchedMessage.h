// SPDX-FileCopyrightText: 2026 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include <cassert>
#include <utility>
#include <vector>
#include "Datatype.h"
#include "Error.h"
#include "Status.h"

namespace mpicpp_lite {

/// Message handle returned from matched probes.
///
/// A valid matched message has been removed from MPI's unexpected-message queue and must be
/// received exactly once.
class MatchedMessage {
public:
    /// Create an empty matched message
    MatchedMessage() = default;

    MatchedMessage(const MatchedMessage &) = delete;
    MatchedMessage & operator=(const MatchedMessage &) = delete;

    /// Move constructor
    MatchedMessage(MatchedMessage && other) noexcept;

    /// Move assignment
    MatchedMessage & operator=(MatchedMessage && other) noexcept;

    /// Check if this object holds a matched message
    explicit operator bool() const;

    /// Get the source of the matched message
    ///
    /// @return Source of the message (i.e. rank ID)
    int source() const;

    /// Get the message tag
    ///
    /// @return Message tag
    int tag() const;

    /// Gets the number of "top level" elements in the matched message
    ///
    /// @tparam T datatype of each receive buffer element
    /// @return The number of "top level" elements
    template <typename T>
    int count() const;

    /// Receive the matched message
    ///
    /// @tparam T C++ type of the data
    /// @param values Variable to receive the data
    /// @param n Number of values to receive
    /// @return `Status` of the operation
    template <typename T>
    Status recv(T * values, int n);

    /// Receive the matched message into a std::vector
    ///
    /// @tparam T C++ type of the data
    /// @param values Variable to receive the data
    /// @return `Status` of the operation
    template <typename T, typename A>
    Status recv(std::vector<T, A> & values);

    /// Receive a matched message without any data
    ///
    /// @return `Status` of the operation
    Status recv();

    MPI_Message &
    native()
    {
        return this->message_;
    }

    const MPI_Message &
    native() const
    {
        return this->message_;
    }

private:
    MatchedMessage(MPI_Message message, Status status, MPI_Comm comm);

    MPI_Message message_ = MPI_MESSAGE_NULL;
    Status status_;
    MPI_Comm comm = MPI_COMM_WORLD;

    friend class Communicator;
};

inline MatchedMessage::MatchedMessage(MPI_Message message, Status status, MPI_Comm comm) :
    message_(message),
    status_(status),
    comm(comm)
{
}

inline MatchedMessage::MatchedMessage(MatchedMessage && other) noexcept :
    message_(std::exchange(other.message_, MPI_MESSAGE_NULL)),
    status_(other.status_),
    comm(other.comm)
{
}

inline MatchedMessage &
MatchedMessage::operator=(MatchedMessage && other) noexcept
{
    if (this != &other) {
        assert(this->message_ == MPI_MESSAGE_NULL);
        this->message_ = std::exchange(other.message_, MPI_MESSAGE_NULL);
        this->status_ = other.status_;
        this->comm = other.comm;
    }
    return *this;
}

inline MatchedMessage::operator bool() const
{
    return this->message_ != MPI_MESSAGE_NULL;
}

inline int
MatchedMessage::source() const
{
    return this->status_.source();
}

inline int
MatchedMessage::tag() const
{
    return this->status_.tag();
}

template <typename T>
inline int
MatchedMessage::count() const
{
    assert(this->message_ != MPI_MESSAGE_NULL);
    int n;
    MPI_Get_count(this->status_, mpi_datatype<T>(), &n);
    return n;
}

template <typename T>
inline Status
MatchedMessage::recv(T * values, int n)
{
    assert(this->message_ != MPI_MESSAGE_NULL);
    assert(n >= 0);
    assert(n == 0 || values != nullptr);

    Status status;
    MPI_CHECK_SELF(MPI_Mrecv(values, n, mpi_datatype<T>(), &this->message_, status));
    return status;
}

template <typename T, typename A>
inline Status
MatchedMessage::recv(std::vector<T, A> & values)
{
    auto size = count<T>();
    assert(size >= 0);
    values.resize(size);
    return recv(values.empty() ? nullptr : values.data(), size);
}

inline Status
MatchedMessage::recv()
{
    assert(this->message_ != MPI_MESSAGE_NULL);

    Status status;
    MPI_CHECK_SELF(MPI_Mrecv(MPI_BOTTOM, 0, MPI_PACKED, &this->message_, status));
    return status;
}

} // namespace mpicpp_lite
