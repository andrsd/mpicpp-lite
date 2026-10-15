Quickstart
==========

``mpicpp-lite`` provides an idiomatic C++20 interface on top of standard MPI. It eliminates boilerplate code such as manually specifying MPI datatypes, buffer lengths, status tracking, and repetitive error-code checking, while maintaining zero runtime overhead.

A Complete "Hello MPI"
----------------------

Here is a complete example demonstrating initialization, rank queries, and collective communication:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <iostream>
   #include <vector>

   namespace mpi = mpicpp_lite;

   int main(int argc, char * argv[])
   {
       // RAII initialization: initializes MPI and automatically finalizes on scope exit
       mpi::Environment env(argc, argv);
       mpi::Communicator comm; // defaults to MPI_COMM_WORLD

       std::cout << "Hello from rank " << comm.rank() 
                 << " of " << comm.size() << std::endl;

       // Broadcast a vector from rank 0 to all other ranks
       std::vector<double> values;
       if (comm.rank() == 0)
           values = { 1.1, 2.2, 3.3, 4.4 };

       comm.broadcast(values, 0);

       // Compute the global sum across all processes
       double local_val = comm.rank() * 2.5;
       double global_sum = 0.0;
       comm.all_reduce(local_val, global_sum, mpi::op::sum<>());

       if (comm.rank() == 0)
           std::cout << "Global sum: " << global_sum << std::endl;

       return 0;
   }

Side-by-Side: Raw C MPI vs. ``mpicpp-lite``
-------------------------------------------

In standard C MPI, every single API call returns an integer error code that production code must rigorously check. Additionally, you must explicitly pass datatypes, buffer sizes, and communicators, and ensure ``MPI_Finalize`` is called on all exit paths.

**Raw C MPI:**

.. code-block:: c

   #include <mpi.h>
   #include <stdio.h>
   #include <stdlib.h>

   int main(int argc, char * argv[])
   {
       int ierr;

       ierr = MPI_Init(&argc, &argv);
       if (ierr != MPI_SUCCESS) {
           fprintf(stderr, "MPI_Init failed\n");
           return 1;
       }

       int rank, size;
       ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       if (ierr != MPI_SUCCESS) {
           MPI_Abort(MPI_COMM_WORLD, ierr);
       }

       ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
       if (ierr != MPI_SUCCESS) {
           MPI_Abort(MPI_COMM_WORLD, ierr);
       }

       // Broadcasting an array:
       int n = 100;
       double data[100];
       if (rank == 0) { /* initialize data */ }
       ierr = MPI_Bcast(data, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       if (ierr != MPI_SUCCESS) {
           MPI_Abort(MPI_COMM_WORLD, ierr);
       }

       // Sum reduction:
       double local_val = rank * 2.5;
       double global_sum;
       ierr = MPI_Allreduce(&local_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       if (ierr != MPI_SUCCESS) {
           MPI_Abort(MPI_COMM_WORLD, ierr);
       }

       ierr = MPI_Finalize();
       if (ierr != MPI_SUCCESS) {
           return 1;
       }
       return 0;
   }

**With ``mpicpp-lite``:**

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <vector>

   namespace mpi = mpicpp_lite;

   int main(int argc, char * argv[])
   {
       // Automatic error checking: throws mpi::Exception on any MPI error
       mpi::Environment env(argc, argv);
       mpi::Communicator comm;

       // Broadcasting a vector: type and size deduced automatically
       std::vector<double> data(100);
       if (comm.rank() == 0) { /* initialize data */ }
       comm.broadcast(data, 0);

       // Sum reduction:
       double local_val = comm.rank() * 2.5;
       double global_sum;
       comm.all_reduce(local_val, global_sum, mpi::op::sum<>());

       return 0;
   } // MPI_Finalize called automatically and safely via RAII

Key Advantages
--------------

- **Automatic Error Checking**: Every underlying MPI call is validated automatically, converting errors into standard C++ exceptions (``mpi::Exception``).
- **Zero Overhead**: Direct inline calls to native MPI primitives without intermediate buffering or abstraction penalties.
- **Type Deduction**: Automatic mapping from C++ types (``int``, ``double``, ``float``, ``char``, etc.) to the corresponding ``MPI_Datatype``.
- **Container Awareness**: First-class support for ``std::vector<T>`` and contiguous memory ranges.
- **Modern C++20**: Uses concepts, transparent comparison functors, and RAII lifetime management.
- **Extensible**: Straightforward registration for user-defined structs and custom reduction operations.
