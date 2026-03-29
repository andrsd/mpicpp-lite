// SPDX-FileCopyrightText: 2023 David Andrs <andrsd@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpi.h"
#include "Enums.h"
#include "Error.h"
#include <vector>

namespace mpicpp_lite {

/// Wrapper around `MPI_Group`
class Group {
public:
    enum ComparisonResult { IDENTICAL = MPI_IDENT, SIMILAR = MPI_SIMILAR, UNEQUAL = MPI_UNEQUAL };

    /// Create an empty group
    Group();

    /// Returns the rank of this process in the given group
    ///
    /// @return Rank of the calling process in group, or MPI_UNDEFINED if the process is not a
    ///         member
    int rank() const;

    /// Returns the size of a group
    ///
    /// @return Number of processes in the group
    int size() const;

    /// Produces a group by reordering an existing group and taking only listed members
    ///
    /// @param ranks Ranks of processes in group to appear in the new group
    /// @return New group derived from this group, in the order defined by `ranks`
    Group include(const std::vector<int> & ranks) const;

    /// Produces a group by reordering an existing group and taking only unlisted members
    ///
    /// @param ranks Array of integer ranks of processes in group not to appear in the new group
    /// @return New group derived from this group, preserving the order defined by group
    Group exclude(const std::vector<int> & ranks) const;

    /// Frees the group
    void free();

    /// Translates the rank of a process in one group to the one in another group
    ///
    /// @param in_rank Rank in this group
    /// @param out_group Destination group
    /// @return Rank in destination group, or `UNDEFINED` when no correspondence exists
    int translate_rank(int in_rank, const Group & out_group) const;

    /// Translates the ranks of processes in one group to those in another group
    ///
    /// @param in_ranks Array of valid ranks in this group
    /// @param out_group Destination group
    /// @return Array of corresponding ranks in destination group, `UNDEFINED` when no
    ///         correspondence exists
    std::vector<int> translate_ranks(const std::vector<int> & in_ranks,
                                     const Group & out_group) const;

public:
    /// Compares two groups
    ///
    /// @param g1 First group
    /// @param g2 Second group
    /// @return `IDENTICAL` if the order and members of the two groups are the same, `SIMILAR` if
    ///         only the members are the same, and `UNEQUAL` otherwise
    static ComparisonResult compare(const Group & g1, const Group & g2);

    /// Produces a group by combining two groups
    ///
    /// @param g1 First group
    /// @param g2 Second group
    /// @return Union group
    static Group join(const Group & g1, const Group & g2);

    /// Produces a group as the intersection of two existing groups
    ///
    /// @param g1 First group
    /// @param g2 Second group
    /// @return Intersection group
    static Group intersection(const Group & g1, const Group & g2);

    /// Makes a group from the difference of two groups
    ///
    /// @param g1 First group
    /// @param g2 Second group
    /// @return Difference group
    static Group difference(const Group & g1, const Group & g2);

private:
    MPI_Group group_;

    friend class Communicator;
    friend class Window;
};

inline Group::Group() : group_(MPI_GROUP_NULL) {}

inline Group
Group::include(const std::vector<int> & ranks) const
{
    Group new_group;
    MPI_CHECK(MPI_Group_incl(this->group_, ranks.size(), ranks.data(), &new_group.group_));
    return new_group;
}

inline Group
Group::exclude(const std::vector<int> & ranks) const
{
    Group new_group;
    MPI_CHECK(MPI_Group_excl(this->group_, ranks.size(), ranks.data(), &new_group.group_));
    return new_group;
}

inline void
Group::free()
{
    MPI_CHECK(MPI_Group_free(&this->group_));
    this->group_ = MPI_GROUP_NULL;
}

inline int
Group::rank() const
{
    int r;
    MPI_CHECK(MPI_Group_rank(this->group_, &r));
    return r;
}

inline int
Group::size() const
{
    int sz;
    MPI_CHECK(MPI_Group_size(this->group_, &sz));
    return sz;
}

inline int
Group::translate_rank(int in_rank, const Group & out_group) const
{
    int out_rank;
    MPI_CHECK(MPI_Group_translate_ranks(this->group_, 1, &in_rank, out_group.group_, &out_rank));
    return out_rank;
}

inline std::vector<int>
Group::translate_ranks(const std::vector<int> & in_ranks, const Group & out_group) const
{
    int n = in_ranks.size();
    std::vector<int> ranks2(n);
    MPI_CHECK(MPI_Group_translate_ranks(this->group_,
                                        n,
                                        in_ranks.data(),
                                        out_group.group_,
                                        ranks2.data()));
    return ranks2;
}

inline Group::ComparisonResult
Group::compare(const Group & g1, const Group & g2)
{
    int result;
    MPI_CHECK(MPI_Group_compare(g1.group_, g2.group_, &result));
    return static_cast<ComparisonResult>(result);
}

inline Group
Group::join(const Group & g1, const Group & g2)
{
    Group new_group;
    MPI_CHECK(MPI_Group_union(g1.group_, g2.group_, &new_group.group_));
    return new_group;
}

inline Group
Group::intersection(const Group & g1, const Group & g2)
{
    Group new_group;
    MPI_CHECK(MPI_Group_intersection(g1.group_, g2.group_, &new_group.group_));
    return new_group;
}

inline Group
Group::difference(const Group & g1, const Group & g2)
{
    Group new_group;
    MPI_CHECK(MPI_Group_difference(g1.group_, g2.group_, &new_group.group_));
    return new_group;
}

} // namespace mpicpp_lite
