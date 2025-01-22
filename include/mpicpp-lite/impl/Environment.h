// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include <vector>

namespace mpicpp_lite {

class Environment {
public:
#if (MPI_VERSION >= 2)
    Environment();
#endif
    Environment(int & argc, char **& argv);
    ~Environment();

    /// Indicates whether MPI was initialized
    ///
    /// @return `true` if MPI is initialized, `false` otherwise
    static bool is_initialized();

    /// Indicates whether MPI was finalized
    ///
    /// @return `true` if MPI is finalized, `false` otherwise
    static bool is_finalized();

private:
    /// Indicates if the environment is initialized
    bool initialized;
};

inline Environment::Environment() : initialized(false)
{
    if (!is_initialized()) {
        MPI_CHECK(MPI_Init(nullptr, nullptr));
        this->initialized = true;
    }
}

inline Environment::Environment(int & argc, char **& argv) : initialized(false)
{
    if (!is_initialized()) {
        MPI_CHECK(MPI_Init(&argc, &argv));
        this->initialized = true;
    }
}

inline Environment::~Environment()
{
    if (this->initialized) {
        if (!is_finalized())
            MPI_CHECK(MPI_Finalize());
    }
}

inline bool
Environment::is_initialized()
{
    int flag;
    MPI_CHECK(MPI_Initialized(&flag));
    return flag != 0;
}

inline bool
Environment::is_finalized()
{
    int flag;
    MPI_CHECK(MPI_Finalized(&flag));
    return flag != 0;
}

//

/// Return the version number of MPI
///
/// @return Tuple with the version and subversion of MPI
inline std::tuple<int, int>
get_mpi_version()
{
    int version, subversion;
    MPI_CHECK(MPI_Get_version(&version, &subversion));
    return std::make_tuple(version, subversion);
}

/// Creates a division of processors in a cartesian grid
///
/// @param n_nodes Number of nodes
/// @param n_dims Number of dimensions
/// @return Vector with the dimensions of the grid
inline std::vector<int>
create_dims(int n_nodes, int n_dims)
{
    std::vector<int> dims(n_dims, 0);
    MPI_CHECK(MPI_Dims_create(n_nodes, n_dims, dims.data()));
    return dims;
}

} // namespace mpicpp_lite
