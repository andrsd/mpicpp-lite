// SPDX-FileCopyrightText: 2026 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "Error.h"
#include <string>
#include <optional>
#include <vector>
#include <format>

namespace mpicpp_lite {

class Info {
public:
    Info();
    Info(const MPI_Info & info);
    ~Info();

    bool is_valid() const;

    void set(const char * key, const char * value);
    void set(const std::string & key, const std::string & value);

    /// Retrieves the value associated with a key in an info object.
    ///
    /// @param key Key
    /// @return Value from the info if defined, `std::nullopt` if not.
    std::optional<std::string> get(const char * key) const;
    std::optional<std::string> get(const std::string & key) const;

    /// Duplicates an info object.
    ///
    /// @return Duplicated info object
    Info duplicate() const;

    /// Delete key
    ///
    /// @param key Key to delete
    void del(const char * key);

    /// Get all keys in the Info object
    ///
    /// @return All keys in the object
    std::vector<std::string> keys() const;

    /// Get value from the info object
    ///
    /// @param key Key to obtain value for
    /// @return Value of the `key`. If `key` does not exist, `out_of_range` exception is thrown.
    std::string operator[](const char * key) const;
    std::string operator[](const std::string & key) const;

    MPI_Info &
    native()
    {
        return this->info_;
    }

    const MPI_Info &
    native() const
    {
        return this->info_;
    }

public:
    static Info env();

private:
    void create();

    MPI_Info info_;
};

inline Info::Info() : info_(MPI_INFO_NULL) {}

inline Info::Info(const MPI_Info & info) : info_(info) {}

inline Info::~Info()
{
    if (is_valid() and this->info_ != MPI_INFO_ENV) {
        MPI_CHECK(MPI_Info_free(&this->info_));
    }
}

inline bool
Info::is_valid() const
{
    return this->info_ != MPI_INFO_NULL;
}

inline void
Info::set(const char * key, const char * value)
{
    if (not is_valid())
        create();

    MPI_CHECK(MPI_Info_set(this->info_, key, value));
}

inline void
Info::set(const std::string & key, const std::string & value)
{
    set(key.c_str(), value.c_str());
}

inline std::optional<std::string>
Info::get(const char * key) const
{
    int len = 0;
    int flag = 0;
    MPI_CHECK(MPI_Info_get_string(this->info_, key, &len, NULL, &flag));
    if (flag == 0)
        return std::nullopt;
    else {
        std::string val(len - 1, '\0');
        MPI_CHECK(MPI_Info_get_string(this->info_, key, &len, &val[0], &flag));
        return val;
    }
}

inline std::optional<std::string>
Info::get(const std::string & key) const
{
    return get(key.c_str());
}

inline Info
Info::duplicate() const
{
    MPI_Info nfo;
    MPI_CHECK(MPI_Info_dup(this->info_, &nfo));
    return { nfo };
}

inline void
Info::del(const char * key)
{
    MPI_CHECK(MPI_Info_delete(this->info_, key));
}

inline std::vector<std::string>
Info::keys() const
{
    int n_keys;
    MPI_CHECK(MPI_Info_get_nkeys(this->info_, &n_keys));
    std::vector<std::string> info_keys;
    if (n_keys > 0) {
        info_keys.reserve(n_keys);
        for (int i = 0; i < n_keys; i++) {
            char val[MPI_MAX_INFO_KEY] = { '\0' };
            MPI_CHECK(MPI_Info_get_nthkey(this->info_, i, &val[0]));
            info_keys.push_back({ val });
        }
    }
    return info_keys;
}

inline std::string
Info::operator[](const char * key) const
{
    auto val = get(key);
    if (val.has_value())
        return val.value();
    else
        throw std::out_of_range(std::format("Trying to get a non-existent key {}", key));
}

inline std::string
Info::operator[](const std::string & key) const
{
    return this->operator[](key.c_str());
}

inline void
Info::create()
{
    MPI_CHECK(MPI_Info_create(&this->info_));
}

inline Info
Info::env()
{
    return { MPI_INFO_ENV };
}

} // namespace mpicpp_lite
