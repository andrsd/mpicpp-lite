// SPDX-FileCopyrightText: 2026 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"

namespace mpicpp_lite {

class Tag {
public:
    constexpr Tag() : value_(0) {}
    explicit constexpr Tag(int tag) : value_(tag) {}

    constexpr int
    value() const
    {
        return this->value_;
    }

    constexpr bool
    operator<(Tag other) const
    {
        return this->value_ < other.value_;
    }

    constexpr bool
    operator==(int other) const
    {
        return this->value_ == other;
    }

    constexpr bool
    operator!=(int other) const
    {
        return this->value_ != other;
    }

private:
    int value_;
};

} // namespace mpicpp_lite
