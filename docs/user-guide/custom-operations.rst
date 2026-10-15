Custom Reductions & Operations
==============================

Beyond standard arithmetic and bitwise reductions, ``mpicpp-lite`` supports finding extrema with their process locations (via ``ValueLoc``) and writing arbitrary user-defined reduction operations for custom types.

.. index::
   single: ValueLoc
   single: minloc
   single: maxloc
   single: extremum with location

Extremum with Location
----------------------

When optimizing, searching, or computing error metrics across a cluster, you often need to know not just the minimum or maximum value, but also *which* process holds that value.

In ``mpicpp-lite``, this is handled using the ``mpi::ValueLoc<T, L>`` struct and the ``mpi::op::minloc<>`` / ``mpi::op::maxloc<>`` operators:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <iostream>

   namespace mpi = mpicpp_lite;

   mpi::Communicator comm;

   // Each process evaluates its local error and attaches its rank
   double local_residual = compute_local_residual();
   mpi::ValueLoc<double> local { local_residual, comm.rank() };
   mpi::ValueLoc<double> global;

   // Global reduction for minimum residual and its rank:
   comm.all_reduce(local, global, mpi::op::minloc<>());

   std::cout << "Lowest residual: " << global.value 
             << " found on rank: "  << global.loc << std::endl;

Tie-Breaking Rule
^^^^^^^^^^^^^^^^^

If two or more processes share the exact same minimum or maximum value, ``minloc`` and ``maxloc`` follow standard MPI semantics: the process with the **smaller rank/location** wins the tie.

Supported Native Types
^^^^^^^^^^^^^^^^^^^^^^

``mpi::ValueLoc`` natively maps to fast MPI hardware-accelerated pair datatypes when using standard numeric types:

- ``ValueLoc<short, int>`` (``MPI_SHORT_INT``)
- ``ValueLoc<int, int>`` (``MPI_2INT``)
- ``ValueLoc<long, int>`` (``MPI_LONG_INT``)
- ``ValueLoc<float, int>`` (``MPI_FLOAT_INT``)
- ``ValueLoc<double, int>`` (``MPI_DOUBLE_INT``)
- ``ValueLoc<long double, int>`` (``MPI_LONG_DOUBLE_INT``)

.. index::
   single: UserOp
   single: IsCommutative
   single: user-defined reductions

User-Defined Reduction Operations
---------------------------------

You can define custom reduction operations for both standard and registered custom datatypes.

Step 1: Declare Commutativity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your operation is mathematically commutative (:math:`x \circ y = y \circ x`), inform the MPI runtime by specializing ``mpi::op::IsCommutative`` with ``std::true_type``. This allows MPI to perform the reduction using faster tree-based algorithms:

.. code-block:: cpp

   namespace mpicpp_lite::op {

   template <>
   struct IsCommutative<sum<Particle>, Particle> : public std::true_type {};

   } // namespace mpicpp_lite::op

Step 2: Specialize `UserOp`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialize your operation template and inherit from ``mpi::op::UserOp``:

.. code-block:: cpp

   namespace mpicpp_lite::op {

   template <>
   struct sum<Particle> : public UserOp<sum<Particle>, Particle> {
       Particle
       operator()(const Particle & a, const Particle & b) const
       {
           Particle result = a;
           result.mass += b.mass;
           result.position[0] += b.position[0];
           result.position[1] += b.position[1];
           result.position[2] += b.position[2];
           return result;
       }
   };

   } // namespace mpicpp_lite::op

Step 3: Execute Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^

Once specialized, your custom operation can be used directly in any collective reduction:

.. code-block:: cpp

   Particle local_particle = get_local_particle();
   Particle total_particle {};

   comm.all_reduce(local_particle, total_particle, mpi::op::sum<>());

   if (comm.rank() == 0) {
       std::cout << "Total mass: " << total_particle.mass << std::endl;
   }
