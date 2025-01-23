// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include "Environment.h"
#include <type_traits>

namespace mpicpp_lite {

namespace op {

/// Template for summation operation on a `T` type
///
/// @tparam T Datatype
template <typename T>
struct sum {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return Sum of `x` and `y`
    const T &
    operator()(const T & x, const T & y) const
    {
        return x + y;
    }
};

// Product

/// Template for product operation on a `T` type
///
/// @tparam T Datatype
template <typename T>
struct prod {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return Product of `x` and `y`
    const T &
    operator()(const T & x, const T & y) const
    {
        return x * y;
    }
};

/// Template for finding maximum on a `T` type
///
/// @tparam T Datatype
template <typename T>
struct max {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return The maximum of `x` and `y`
    const T &
    operator()(const T & x, const T & y) const
    {
        return x < y ? y : x;
    }
};

/// Template for finding minimum on a `T` type
///
/// @tparam T Datatype
template <typename T>
struct min {
    /// Call operator
    ///
    /// @param x First operand
    /// @param y Second operand
    /// @return The minimum of `x` and `y`
    const T &
    operator()(const T & x, const T & y) const
    {
        return x < y ? x : y;
    }
};

/// Template for logical AND on a `T` type
///
/// @tparam T Datatype
template <typename T>
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

/// Template for logical OR on a `T` type
///
/// @tparam T Datatype
template <typename T>
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

/// Template for logical XOR on a `T` type
///
/// @tparam T Datatype
template <typename T>
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

/// Determine if a function object type is commutative.
///
/// This trait determines if an operation `Op` is commutative when applied to values of type `T`.
/// Parallel operations such as reduce can be implemented more efficiently with commutative
/// operations. To mark an operation as commutative, users should specialize `IsCommutative` and
/// derive from the class `std::true_type`.
template <typename Op, typename T>
struct IsCommutative : public std::false_type {};

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
template <typename T>
struct Operation<sum<T>, T> {
    using is_native = typename std::conditional<
        std::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value,
        std::true_type,
        std::false_type>::type;

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
template <typename T>
struct Operation<prod<T>, T> {
    using is_native = typename std::conditional<
        std::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value,
        std::true_type,
        std::false_type>::type;

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
template <typename T>
struct Operation<max<T>, T> {
    using is_native = typename std::conditional<
        std::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value,
        std::true_type,
        std::false_type>::type;

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
template <typename T>
struct Operation<min<T>, T> {
    using is_native = typename std::conditional<
        std::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value,
        std::true_type,
        std::false_type>::type;

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
template <typename T>
struct Operation<logical_and<T>, T> {
    using is_native = typename std::
        conditional<std::is_integral<T>::value, std::true_type, std::false_type>::type;

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
template <typename T>
struct Operation<logical_or<T>, T> {
    using is_native = typename std::
        conditional<std::is_integral<T>::value, std::true_type, std::false_type>::type;

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
template <typename T>
struct Operation<logical_xor<T>, T> {
    using is_native = typename std::
        conditional<std::is_integral<T>::value, std::true_type, std::false_type>::type;

    /// Call operator
    ///
    /// @return MPI operation for logical XOR
    static MPI_Op
    op()
    {
        return MPI_LXOR;
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
    /// Get the `MPI_Op` for the user-defined operation
    static MPI_Op
    op()
    {
        static auto mop = create(&UserOp<Op, T>::perform, IsCommutative<Op, T>::value);
        return mop;
    }

private:
    static void
    perform(void * a, void * b, int * len, MPI_Datatype *)
    {
        T * invec = static_cast<T *>(a);
        T * outvec = static_cast<T *>(b);
        Op op;
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
