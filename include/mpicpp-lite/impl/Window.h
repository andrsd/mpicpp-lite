// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include "Operation.h"
#include "Datatype.h"
#include "Group.h"
#include <vector>

namespace mpicpp_lite {

enum class Lock {
    SHARED = MPI_LOCK_SHARED,
    EXCLUSIVE = MPI_LOCK_EXCLUSIVE
};

class Window {
public:
    Window();
    ~Window();

    /// Attach memory to a dynamic window
    ///
    /// @param base Initial address of memory to be attached
    /// @param size Size of memory to be attached (in bytes)
    void attach(void * base, MPI_Aint size) const;

    /// Attach C++ object to a dynamic window
    ///
    /// @tparam T Type of the data
    /// @param base Object to be attached
    template <typename T>
    void attach(T & base) const;

    /// Attach std:vector to a dynamic window
    ///
    /// @tparam T Type of the data
    /// @param base Vector to be attached
    template <typename T>
    void attach(std::vector<T> & base) const;

    /// Detach memory from a dynamic window
    ///
    /// @param base initial address of memory to be detached
    void detach(const void * base) const;

    template <typename T>
    void detach(const T & base) const;

    template <typename T>
    void detach(const std::vector<T> & base) const;

    /// Complete all outstanding RMA operations at the given target.
    ///
    /// @param rank Rank of window
    void flush(int rank) const;

    /// Complete all outstanding RMA operations at all targets
    void flush_all() const;

    /// Complete locally all outstanding RMA operations at the given target
    ///
    /// @param rank Rank of window
    void flush_local(int rank) const;

    /// Complete locally all outstanding RMA operations at all targets
    void flush_local_all() const;

    /// Get the MPI Group of the window object
    ///
    /// @return The group of the window
    Group group() const;

    /// Start an RMA access epoch for MPI
    ///
    /// @param group Group of target processes
    /// @param assert Used to optimize this call
    void start(MPI_Group group, int assert = 0) const;

    /// Set the print name for an MPI RMA window
    ///
    /// @param name The character string which is remembered as the name
    void set_name(const char * name) const;

    /// Completes an RMA operations begun after an `start`
    void complete() const;

    void free();

    /// Start an RMA exposure epoch
    ///
    /// @param group Group of target processes
    /// @param assert Used to optimize this call
    void post(MPI_Group group, int assert = 0) const;

    void lock(Lock lock_type, int rank, int assert = 0) const;

    void lock_all(int assert = 0) const;

    /// Get the print name associated with the MPI RMA window
    ///
    /// @return The name of the window
    std::string name() const;

    /// Synchronize public and private copies of the given window
    void sync() const;

    /// Test whether an RMA exposure epoch has completed
    ///
    /// @return `true` if the epoch has completed, `false` otherwise
    bool test() const;

    /// Completes an RMA access epoch at the target process
    ///
    /// @param rank Rank of target
    void unlock(int rank) const;

    /// Completes an RMA access epoch at all processes on the given window.
    void unlock_all() const;

    /// Completes an RMA exposure epoch begun with `post`
    void wait() const;

    /// Get data from a memory window on a remote process
    ///
    /// @tparam T Type of the data
    /// @param origin_addr Address of the buffer in which to receive the data
    /// @param origin_count Number of entries in origin buffer
    /// @param target_rank Rank of target
    /// @param target_disp Displacement from start of window to target buffer
    /// @param target_count Number of entries in target buffer
    template <typename T>
    void get(T * origin_addr,
             int origin_count,
             int target_rank,
             MPI_Aint target_disp,
             int target_count) const;

    template <typename T>
    void get(std::vector<T> & origin, int target_rank) const;

    /// Put data into a memory window on a remote process
    ///
    /// @tparam T Type of the data
    /// @param origin_addr Initial address of buffer
    /// @param origin_count Number of entries in origin buffer
    /// @param target_rank Rank of target
    /// @param target_disp Displacement from start of window to target buffer
    /// @param target_count Number of entries in target buffer
    template <typename T>
    void put(const T * origin_addr,
             int origin_count,
             int target_rank,
             MPI_Aint target_disp,
             int target_count) const;

    /// Perform an atomic, one-sided read-and-accumulate operation.
    ///
    /// @tparam T Type of the data
    /// @tparam Op Type of the operation
    /// @param origin_addr Initial address of buffer
    /// @param origin_count Number of entries in buffer
    /// @param result_addr Initial address of result buffer
    /// @param result_count Number of entries in result buffer
    /// @param target_rank Rank of target
    /// @param target_disp Displacement from start of window to target buffer
    /// @param target_count Number of entries in target buffer
    template <typename T, typename Op>
    void accumulate(const T * origin_addr,
                    int origin_count,
                    T * result_addr,
                    int result_count,
                    int target_rank,
                    MPI_Aint target_disp,
                    int target_count,
                    Op) const;

private:
    MPI_Win win;

public:
    static Window create(void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm);

    template <typename T>
    static Window create(std::vector<T> & base, MPI_Info info, MPI_Comm comm);

    static Window create_dynamic(MPI_Info info, MPI_Comm comm);
};

inline Window::Window() : win(MPI_WIN_NULL) {}

inline Window::~Window()
{
    if (win != MPI_WIN_NULL)
        free();
}

inline void
Window::attach(void * base, MPI_Aint size) const
{
    MPI_CHECK(MPI_Win_attach(this->win, base, size));
}

template <typename T>
inline void
Window::attach(T & base) const
{
    attach(&base, sizeof(T));
}

template <typename T>
inline void
Window::attach(std::vector<T> & base) const
{
    attach(base.data(), base.size() * sizeof(T));
}

inline void
Window::detach(const void * base) const
{
    MPI_CHECK(MPI_Win_detach(this->win, base));
}

template <typename T>
inline void
Window::detach(const T & base) const
{
    detach(&base, sizeof(T));
}

template <typename T>
inline void
Window::detach(const std::vector<T> & base) const
{
    detach(base.data(), base.size() * sizeof(T));
}

inline void
Window::free()
{
    MPI_CHECK(MPI_Win_free(&this->win));
    this->win = MPI_WIN_NULL;
}

inline void
Window::lock(Lock lock_type, int rank, int assert) const
{
    MPI_CHECK(MPI_Win_lock(static_cast<int>(lock_type), rank, assert, this->win));
}

inline void
Window::lock_all(int assert) const
{
    MPI_CHECK(MPI_Win_lock_all(assert, this->win));
}

inline void
Window::flush(int rank) const
{
    MPI_CHECK(MPI_Win_flush(rank, this->win));
}

inline void
Window::flush_all() const
{
    MPI_CHECK(MPI_Win_flush_all(this->win));
}

inline void
Window::flush_local(int rank) const
{
    MPI_CHECK(MPI_Win_flush_local(rank, this->win));
}

inline void
Window::flush_local_all() const
{
    MPI_CHECK(MPI_Win_flush_local_all(this->win));
}

inline Group
Window::group() const
{
    MPI_Group g;
    MPI_CHECK(MPI_Win_get_group(this->win, &g));
    return { g };
}

inline void
Window::start(MPI_Group group, int assert) const
{
    MPI_CHECK(MPI_Win_start(group, assert, this->win));
}

inline void
Window::set_name(const char * name) const
{
    MPI_CHECK(MPI_Win_set_name(this->win, name));
}

inline void
Window::complete() const
{
    MPI_CHECK(MPI_Win_complete(this->win));
}

inline std::string
Window::name() const
{
    char nm[MPI_MAX_OBJECT_NAME];
    int len;
    MPI_CHECK(MPI_Win_get_name(this->win, nm, &len));
    return std::string(nm);
}

inline void
Window::post(MPI_Group group, int assert) const
{
    MPI_CHECK(MPI_Win_post(group, assert, this->win));
}

inline bool
Window::test() const
{
    int flag;
    MPI_CHECK(MPI_Win_test(this->win, &flag));
    return flag != 0;
}

inline void
Window::unlock(int rank) const
{
    MPI_CHECK(MPI_Win_unlock(rank, this->win));
}

inline void
Window::unlock_all() const
{
    MPI_CHECK(MPI_Win_unlock_all(this->win));
}

inline void
Window::wait() const
{
    MPI_CHECK(MPI_Win_wait(this->win));
}

inline void
Window::sync() const
{
    MPI_CHECK(MPI_Win_sync(this->win));
}

inline Window
Window::create(void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm)
{
    Window w;
    MPI_CHECK(MPI_Win_create(base, size, disp_unit, info, comm, &w.win));
    return w;
}

template <typename T>
inline Window
Window::create(std::vector<T> & base, MPI_Info info, MPI_Comm comm)
{
    return create(base.data(), base.size() * sizeof(T), sizeof(T), info, comm);
}

inline Window
Window::create_dynamic(MPI_Info info, MPI_Comm comm)
{
    Window w;
    MPI_CHECK(MPI_Win_create_dynamic(info, comm, &w.win));
    return w;
}

template <typename T>
inline void
Window::get(T * origin_addr,
            int origin_count,
            int target_rank,
            MPI_Aint target_disp,
            int target_count) const
{
    MPI_CHECK(MPI_Get(origin_addr,
                      origin_count,
                      mpi_datatype<T>(),
                      target_rank,
                      target_disp,
                      target_count,
                      mpi_datatype<T>(),
                      this->win));
}

template <typename T>
inline void
Window::get(std::vector<T> & origin, int target_rank) const
{
    get(origin.data(), origin.size(), target_rank, 0, origin.size());
}

template <typename T>
inline void
Window::put(const T * origin_addr,
         int origin_count,
         int target_rank,
         MPI_Aint target_disp,
         int target_count) const
{
    MPI_CHECK(MPI_Put(origin_addr,
                      origin_count,
                      mpi_datatype<T>(),
                      target_rank,
                      target_disp,
                      target_count,
                      mpi_datatype<T>(),
                      this->win));
}

template <typename T, typename Op>
inline void
Window::accumulate(const T * origin_addr,
                   int origin_count,
                   T * result_addr,
                   int result_count,
                   int target_rank,
                   MPI_Aint target_disp,
                   int target_count,
                   Op) const
{
    MPI_CHECK(MPI_Get_accumulate(origin_addr,
                                 origin_count,
                                 mpi_datatype<T>(),
                                 result_addr,
                                 result_count,
                                 mpi_datatype<T>(),
                                 target_rank,
                                 target_disp,
                                 target_count,
                                 mpi_datatype<T>(),
                                 op::provider<T, Op, op::Operation<Op, T>::is_native::value>::op(),
                                 this->win));
}

} // namespace mpicpp_lite
