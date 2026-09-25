// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include <algorithm>
#include <type_traits>
#include <concepts>

namespace mpicpp_lite {

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

namespace op {

/// Template for summation operation on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct sum {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return Sum of `x` and `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x + y;
    }
};

// Transparent specialization for void
template <>
struct sum<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const -> decltype(std::forward<U>(x) + std::forward<V>(y))
    {
        return std::forward<U>(x) + std::forward<V>(y);
    }

    using is_transparent = void;
};

// Product

/// Template for product operation on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct prod {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return Product of `x` and `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x * y;
    }
};

// Transparent specialization for void
template <>
struct prod<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const -> decltype(std::forward<U>(x) * std::forward<V>(y))
    {
        return std::forward<U>(x) * std::forward<V>(y);
    }

    using is_transparent = void;
};

/// Template for finding maximum on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct max {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return The maximum of `x` and `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x < y ? y : x;
    }
};

// Transparent specialization for void
template <>
struct max<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const
        -> decltype(std::forward<U>(x) < std::forward<V>(y) ? std::forward<V>(y)
                                                            : std::forward<U>(x))
    {
        return std::forward<U>(x) < std::forward<V>(y) ? std::forward<V>(y) : std::forward<U>(x);
    }

    using is_transparent = void;
};

/// Template for finding minimum on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct min {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return The minimum of `x` and `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x < y ? x : y;
    }
};

// Transparent specialization for void
template <>
struct min<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const
        -> decltype(std::forward<U>(x) < std::forward<V>(y) ? std::forward<U>(x)
                                                            : std::forward<V>(y))
    {
        return std::forward<U>(x) < std::forward<V>(y) ? std::forward<U>(x) : std::forward<V>(y);
    }

    using is_transparent = void;
};

/// Template for logical AND on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct logical_and {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return `x` AND `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x && y;
    }
};

// Transparent specialization for void
template <>
struct logical_and<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const -> decltype(std::forward<U>(x) && std::forward<V>(y))
    {
        return std::forward<U>(x) && std::forward<V>(y);
    }

    using is_transparent = void;
};

/// Template for logical OR on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct logical_or {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return `x` OR `y`
    T
    operator()(const T & x, const T & y) const
    {
        return x || y;
    }
};

// Transparent specialization for void
template <>
struct logical_or<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const -> decltype(std::forward<U>(x) || std::forward<V>(y))
    {
        return std::forward<U>(x) || std::forward<V>(y);
    }

    using is_transparent = void;
};

/// Template for logical XOR on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct logical_xor {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return `x` XOR `y`
    T
    operator()(const T & x, const T & y) const
    {
        return !x != !y;
    }
};

// Transparent specialization for void
template <>
struct logical_xor<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && y) const -> decltype(!std::forward<U>(x) != !std::forward<V>(y))
    {
        return !std::forward<U>(x) != !std::forward<V>(y);
    }

    using is_transparent = void;
};

/// Template for replace on a `T` type
///
/// @tparam T Datatype
template <typename T = void>
struct replace {
    T
    operator()(const T & x, const T & /*y*/) const
    {
        return x;
    }
};

// Transparent specialization for void
template <>
struct replace<void> {
    template <typename U, typename V>
    constexpr auto
    operator()(U && x, V && /*y*/) const -> decltype(std::forward<U>(x))
    {
        return std::forward<U>(x);
    }

    using is_transparent = void;
};

/// Determine if a function object type is commutative.
///
/// This trait determines if an operation `Op` is commutative when applied to values of type `T`.
/// Parallel operations such as reduce can be implemented more efficiently with commutative
/// operations. To mark an operation as commutative, users should specialize `IsCommutative` and
/// derive from the class `std::true_type`.
template <typename Op, typename T>
struct IsCommutative : public std::false_type {};

/// Map potentially transparent operation template to its specialized type `T` representation.
template <typename Op, typename T>
struct MapOp {
    using type = Op;
};

template <typename T>
struct MapOp<sum<void>, T> {
    using type = sum<T>;
};

template <typename T>
struct MapOp<prod<void>, T> {
    using type = prod<T>;
};

template <typename T>
struct MapOp<max<void>, T> {
    using type = max<T>;
};

template <typename T>
struct MapOp<min<void>, T> {
    using type = min<T>;
};

template <typename T>
struct MapOp<logical_and<void>, T> {
    using type = logical_and<T>;
};

template <typename T>
struct MapOp<logical_or<void>, T> {
    using type = logical_or<T>;
};

template <typename T>
struct MapOp<logical_xor<void>, T> {
    using type = logical_xor<T>;
};

template <typename T>
struct MapOp<replace<void>, T> {
    using type = replace<T>;
};

/// Template for MPI operation `Op` on a `T` type
///
/// @tparam Op Operation
/// @tparam T Datatype
template <typename Op, typename T>
struct Operation {
    /// Determines if the operation is a native MPI operation. User-defined operations
    /// must have this trait set to `std::false_type` to work correctly
    using is_native = typename std::false_type;
};

/// Template for summation operation on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::sum<void>> || std::same_as<Op, op::sum<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for sumation
    static MPI_Op
    op()
    {
        return MPI_SUM;
    }
};

/// Template for product operation on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::prod<void>> || std::same_as<Op, op::prod<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for product
    static MPI_Op
    op()
    {
        return MPI_PROD;
    }
};

/// Template for finding maximum on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::max<void>> || std::same_as<Op, op::max<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for finding maximum
    static MPI_Op
    op()
    {
        return MPI_MAX;
    }
};

/// Template for finding minimum on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::min<void>> || std::same_as<Op, op::min<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for finding minimum
    static MPI_Op
    op()
    {
        return MPI_MIN;
    }
};

/// Template for logical AND on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::logical_and<void>> || std::same_as<Op, op::logical_and<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation logical AND
    static MPI_Op
    op()
    {
        return MPI_LAND;
    }
};

/// Template for logical OR on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::logical_or<void>> || std::same_as<Op, op::logical_or<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for logical OR
    static MPI_Op
    op()
    {
        return MPI_LOR;
    }
};

/// Template for logical XOR on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::logical_xor<void>> || std::same_as<Op, op::logical_xor<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for logical XOR
    static MPI_Op
    op()
    {
        return MPI_LXOR;
    }
};

/// Template for replace on a `T` type
///
/// @tparam T Datatype
template <typename Op, Numeric T>
    requires std::same_as<Op, op::replace<void>> || std::same_as<Op, op::replace<T>>
struct Operation<Op, T> {
    using is_native = std::true_type;

    /// Call operator
    ///
    /// @return MPI operation for logical XOR
    static MPI_Op
    op()
    {
        return MPI_REPLACE;
    }
};

/// Create a new user-defined operation
///
/// @param user_fn Function that implements the operation
/// @param commute `true` if the operation is commutative, `false` otherwise
/// @return User-defined operation
inline MPI_Op
create(MPI_User_function * user_fn, bool commute)
{
    MPI_Op op;
    MPI_CHECK(MPI_Op_create(user_fn, commute, &op));
    return op;
}

/// User-defined operation `Op` on datatype `T`
///
/// @tparam Op Operation
/// @tparam T Datatype
template <typename Op, typename T>
struct UserOp {
    using MappedOp = typename MapOp<Op, T>::type;

    /// Get the `MPI_Op` for the user-defined operation
    static MPI_Op
    op()
    {
        static auto mop = create(&UserOp<Op, T>::perform, IsCommutative<MappedOp, T>::value);
        return mop;
    }

private:
    static void
    perform(void * a, void * b, int * len, MPI_Datatype *)
    {
        T * invec = static_cast<T *>(a);
        T * outvec = static_cast<T *>(b);
        MappedOp op;
        std::transform(invec, invec + *len, outvec, outvec, op);
    }
};

/// Provides the `MPI_Op` for a given operation `Op` on datatype `T`
///
/// @tparam T Datatype
/// @tparam Op Operation
/// @tparam NATIVE `true` if `Op` is a native MPI operation, `false` for user-defined operations
template <typename T, typename Op, bool NATIVE>
struct provider {
    /// Provides the `MPI_Op` for a given operation `Op` on datatype `T`
    static MPI_Op op();
};

template <typename T, typename Op>
struct provider<T, Op, true> {
    static MPI_Op
    op()
    {
        return op::Operation<Op, T>::op();
    }
};

template <typename T, typename Op>
struct provider<T, Op, false> {
    static MPI_Op
    op()
    {
        return op::UserOp<Op, T>::op();
    }
};

} // namespace op

} // namespace mpicpp_lite
