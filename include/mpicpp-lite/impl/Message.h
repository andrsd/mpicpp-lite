// SPDX-FileCopyrightText: 2026 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"

namespace mpicpp_lite {

class Message {
public:
    /// Construct empty `Message` object
    Message();

    bool is_valid() const;

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
    MPI_Message message_ = MPI_MESSAGE_NULL;
};

inline Message::Message() : message_(MPI_MESSAGE_NULL)
{
    static_assert(sizeof(Message) == sizeof(MPI_Message),
                  "Size of `Message` must match `MPI_Message`");
}

inline bool
Message::is_valid() const
{
    return this->message_ != MPI_MESSAGE_NULL;
}

} // namespace mpicpp_lite
