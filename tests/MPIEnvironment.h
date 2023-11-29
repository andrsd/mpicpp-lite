#pragma once

#include "mpicpp-lite/mpicpp-lite.h"

namespace mpi = mpicpp_lite;

class MPIEnvironment : public ::testing::Environment {
public:
    void
    SetUp() override
    {
        this->env = new mpi::Environment();
    }

    void
    TearDown() override
    {
        delete this->env;
        this->env = nullptr;
    }

private:
    mpi::Environment * env = nullptr;
};
