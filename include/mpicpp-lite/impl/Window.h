// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Error.h"
#include "Operation.h"
#include "Datatype.h"
#include "Group.h"
#include "Info.h"
#include <vector>
#include <optional>
#include <utility>
#include <type_traits>
#include <span>

namespace mpicpp_lite {

enum class Lock {
    //
    SHARED = MPI_LOCK_SHARED,
    EXCLUSIVE = MPI_LOCK_EXCLUSIVE
};

class Window {
public:
    class Key {
    public:
        constexpr Key() : value_(0) {}
        explicit constexpr Key(int val) : value_(val) {}

        constexpr int
        value() const
        {
            return this->value_;
        }

        constexpr bool
        operator==(Key other) const
        {
            return this->value_ == other.value_;
        }

        constexpr bool
        operator==(int other) const
        {
            return this->value_ == other;
        }

        constexpr bool
        operator!=(Key other) const
        {
            return this->value_ != other.value_;
        }

        constexpr bool
        operator!=(int other) const
        {
            return this->value_ != other;
        }

    private:
        int value_;
    };

    Window();

    /// Set new configuration hints for an RMA window
    ///
    /// @param info Info object containing the new hints
    void set_info(const Info & info) const;

    /// Get active hints associated with an RMA window
    ///
    /// @return Info object containing the hints
    Info info() const;

    /// Retrieves attribute value by key on a window
    ///
    /// @param key Key of the attribute to get
    /// @return Attribute value, if found, otherwise `std::nullopt`
    template <typename T>
        requires std::copyable<T>
    std::optional<T> attr(Key key) const;

    /// Stores attribute value associated with a key on a window
    template <typename T>
        requires std::copyable<T>
    void set_attr(Key key, const T & val) const;

    /// Deletes an attribute value associated with a key on a window
    ///
    /// @param key Key of the attribute to delete
    void delete_attr(Key key) const;

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
    void start(Group group, int assert = 0) const;

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
    void post(Group group, int assert = 0) const;

    void lock(Lock lock_type, int rank, int assert = 0) const;

    void lock_all(int assert = 0) const;

    /// Get the print name associated with the MPI RMA window
    ///
    /// @return The name of the window
    std::string name() const;

    /// Synchronize public and private copies of the given window
    void sync() const;

    /// Check if window is valid
    operator bool() const;

    /// Check if window is valid
    bool is_valid() const;

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
    MPI_Win win_;

public:
    /// Creates a keyval for RMA windows
    static inline Key
    create_key(void * extra_state = nullptr)
    {
        int keyval = MPI_KEYVAL_INVALID;
        MPI_CHECK(MPI_Win_create_keyval(MPI_WIN_NULL_COPY_FN,
                                        MPI_WIN_NULL_DELETE_FN,
                                        &keyval,
                                        extra_state));
        Environment::win_key_vals_.push_back(keyval);
        return Key { keyval };
    }

    /// Allocate memory and create an MPI RMA window
    ///
    /// @param size Size of the window in bytes
    /// @param disp_unit Displacement unit for window, in bytes
    /// @param info Info object containing hints
    /// @param comm Communicator
    /// @return Pair containing the allocated window and the base pointer to the allocated memory
    static std::pair<Window, void *>
    allocate(MPI_Aint size, int disp_unit, Info info, MPI_Comm comm);

    /// Allocate memory and create an MPI RMA window for a specific type
    ///
    /// @tparam T Type of elements to allocate
    /// @param n Number of elements to allocate
    /// @param info Info object containing hints
    /// @param comm Communicator
    /// @return Pair containing the allocated window and the typed pointer to the allocated memory
    template <typename T>
    static std::pair<Window, T *> allocate(MPI_Aint n, Info info, MPI_Comm comm);

    static Window create(void * base, MPI_Aint size, int disp_unit, Info info, MPI_Comm comm);

    template <typename T>
    static Window create(std::span<T> base, Info info, MPI_Comm comm);

    static Window create_dynamic(Info info, MPI_Comm comm);
};

inline Window::Window() : win_(MPI_WIN_NULL) {}

inline void
Window::attach(void * base, MPI_Aint size) const
{
    MPI_CHECK(MPI_Win_attach(this->win_, base, size));
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
    MPI_CHECK(MPI_Win_detach(this->win_, base));
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
    MPI_CHECK(MPI_Win_free(&this->win_));
    this->win_ = MPI_WIN_NULL;
}

inline void
Window::lock(Lock lock_type, int rank, int assert) const
{
    MPI_CHECK(MPI_Win_lock(static_cast<int>(lock_type), rank, assert, this->win_));
}

inline void
Window::lock_all(int assert) const
{
    MPI_CHECK(MPI_Win_lock_all(assert, this->win_));
}

inline void
Window::flush(int rank) const
{
    MPI_CHECK(MPI_Win_flush(rank, this->win_));
}

inline void
Window::flush_all() const
{
    MPI_CHECK(MPI_Win_flush_all(this->win_));
}

inline void
Window::flush_local(int rank) const
{
    MPI_CHECK(MPI_Win_flush_local(rank, this->win_));
}

inline void
Window::flush_local_all() const
{
    MPI_CHECK(MPI_Win_flush_local_all(this->win_));
}

inline Group
Window::group() const
{
    Group g;
    MPI_CHECK(MPI_Win_get_group(this->win_, &g.group_));
    return Group(g);
}

inline void
Window::start(Group group, int assert) const
{
    MPI_CHECK(MPI_Win_start(group.group_, assert, this->win_));
}

inline void
Window::set_name(const char * name) const
{
    MPI_CHECK(MPI_Win_set_name(this->win_, name));
}

inline void
Window::complete() const
{
    MPI_CHECK(MPI_Win_complete(this->win_));
}

inline std::string
Window::name() const
{
    char nm[MPI_MAX_OBJECT_NAME];
    int len;
    MPI_CHECK(MPI_Win_get_name(this->win_, nm, &len));
    return std::string(nm);
}

inline void
Window::post(Group group, int assert) const
{
    MPI_CHECK(MPI_Win_post(group.group_, assert, this->win_));
}

inline bool
Window::test() const
{
    int flag;
    MPI_CHECK(MPI_Win_test(this->win_, &flag));
    return flag != 0;
}

inline void
Window::unlock(int rank) const
{
    MPI_CHECK(MPI_Win_unlock(rank, this->win_));
}

inline void
Window::unlock_all() const
{
    MPI_CHECK(MPI_Win_unlock_all(this->win_));
}

inline void
Window::wait() const
{
    MPI_CHECK(MPI_Win_wait(this->win_));
}

inline void
Window::sync() const
{
    MPI_CHECK(MPI_Win_sync(this->win_));
}

inline Window::operator bool() const
{
    return is_valid();
}

inline bool
Window::is_valid() const
{
    return this->win_ != MPI_WIN_NULL;
}

inline Window
Window::create(void * base, MPI_Aint size, int disp_unit, Info info, MPI_Comm comm)
{
    Window w;
    MPI_CHECK(MPI_Win_create(base, size, disp_unit, info.native(), comm, &w.win_));
    return w;
}

template <typename T>
inline Window
Window::create(std::span<T> base, Info info, MPI_Comm comm)
{
    return create(base.data(), base.size() * sizeof(T), sizeof(T), info.native(), comm);
}

inline Window
Window::create_dynamic(Info info, MPI_Comm comm)
{
    Window w;
    MPI_CHECK(MPI_Win_create_dynamic(info.native(), comm, &w.win_));
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
                      this->win_));
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
                      this->win_));
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
                                 this->win_));
}

inline void
Window::set_info(const Info & info) const
{
    MPI_CHECK(MPI_Win_set_info(this->win_, info.native()));
}

inline Info
Window::info() const
{
    MPI_Info info;
    MPI_CHECK(MPI_Win_get_info(this->win_, &info));
    return Info(info);
}

template <typename T>
    requires std::copyable<T>
inline std::optional<T>
Window::attr(Key key) const
{
    void * val = nullptr;
    int flag = 0;
    MPI_CHECK(MPI_Win_get_attr(this->win_, key.value(), &val, &flag));
    if (flag) {
        if constexpr (std::is_pointer_v<T>) {
            return reinterpret_cast<T>(val);
        }
        else {
            return *reinterpret_cast<T *>(val);
        }
    }
    else
        return std::nullopt;
}

template <typename T>
    requires std::copyable<T>
inline void
Window::set_attr(Key key, const T & val) const
{
    MPI_CHECK(MPI_Win_set_attr(this->win_, key.value(), const_cast<T *>(&val)));
}

inline void
Window::delete_attr(Key key) const
{
    MPI_CHECK(MPI_Win_delete_attr(this->win_, key.value()));
}

inline std::pair<Window, void *>
Window::allocate(MPI_Aint size, int disp_unit, Info info, MPI_Comm comm)
{
    Window w;
    void * baseptr = nullptr;
    MPI_CHECK(MPI_Win_allocate(size, disp_unit, info.native(), comm, &baseptr, &w.win_));
    return { w, baseptr };
}

template <typename T>
inline std::pair<Window, T *>
Window::allocate(MPI_Aint n, Info info, MPI_Comm comm)
{
    auto [w, baseptr] = allocate(n * sizeof(T), sizeof(T), info, comm);
    return { w, static_cast<T *>(baseptr) };
}

constexpr Window::Key win_base { MPI_WIN_BASE };
constexpr Window::Key win_size { MPI_WIN_SIZE };
constexpr Window::Key win_disp_unit { MPI_WIN_DISP_UNIT };
constexpr Window::Key win_create_flavor { MPI_WIN_CREATE_FLAVOR };
constexpr Window::Key win_model { MPI_WIN_MODEL };

} // namespace mpicpp_lite
