// SPDX-FileCopyrightText: 2026 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

namespace mpicpp_lite {

/// Value and location pair for MINLOC / MAXLOC operations
///
/// @tparam T Value type
/// @tparam L Location/index type (defaults to int)
template <typename T, typename L = int>
struct ValueLoc {
    T value {};
    L loc {};

    bool operator==(const ValueLoc &) const = default;
};

} // namespace mpicpp_lite
