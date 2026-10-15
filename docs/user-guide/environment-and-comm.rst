Environment & Communicators
============================

.. index::
   single: Environment
   single: initialization

Environment Lifecycle
---------------------

``mpicpp-lite`` encapsulates MPI initialization and finalization via the ``Environment`` class. It employs the RAII (Resource Acquisition Is Initialization) pattern to guarantee that ``MPI_Finalize`` is always invoked upon scope exit.

Basic Initialization
^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"

   namespace mpi = mpicpp_lite;

   int main(int argc, char * argv[])
   {
       mpi::Environment env(argc, argv);
       // MPI is now initialized and ready for communication

       return 0;
   } // MPI_Finalize is automatically called when `env` leaves scope

.. index::
   single: ThreadLevel
   single: multi-threading

Thread-Level Support
^^^^^^^^^^^^^^^^^^^^

If your application uses multithreading alongside MPI, you can request specific thread levels using ``ThreadLevel``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Environment env(argc, argv, mpi::ThreadLevel::MULTIPLE);

   if (env.thread_level() == mpi::ThreadLevel::MULTIPLE) {
       // Multi-threaded MPI calls are supported
   }

Available levels correspond to standard MPI threading modes:

- ``ThreadLevel::SINGLE``: Only one thread will execute.
- ``ThreadLevel::FUNNELED``: Process may be multithreaded, but only the main thread makes MPI calls.
- ``ThreadLevel::SERIALIZED``: Multithreaded, but only one thread makes MPI calls at a time.
- ``ThreadLevel::MULTIPLE``: Multiple threads may make MPI calls simultaneously.

.. index::
   single: time
   single: timing; wall clock

Wall-Clock Timing
^^^^^^^^^^^^^^^^^

To benchmark code, ``mpicpp-lite`` provides high-resolution wall-clock timing functions:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   double start = mpi::time();
   // ... perform work ...
   double end = mpi::time();
   double elapsed = end - start;

.. index::
   single: Communicator

Working with Communicators
--------------------------

The ``Communicator`` class represents an MPI communicator. By default, constructing a ``Communicator`` wraps ``MPI_COMM_WORLD``.

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Communicator comm;

   int rank = comm.rank(); // Equivalent to MPI_Comm_rank
   int size = comm.size(); // Equivalent to MPI_Comm_size

.. index::
   single: Communicator::split
   single: split

Subdividing Communicators
^^^^^^^^^^^^^^^^^^^^^^^^^

You can subdivide an existing communicator into smaller disjoint sub-communicators using ``split()``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   // Split processes into even and odd groups
   int color = (comm.rank() % 2 == 0) ? 0 : 1;
   int key = comm.rank();

   mpi::Communicator sub_comm = comm.split(color, key);
   std::cout << "Global rank " << comm.rank() 
             << " has sub-comm rank " << sub_comm.rank() << std::endl;

.. index::
   single: Communicator::split_type
   single: shared memory communicator

Shared-Memory Sub-Communicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also split communicators by node topology to create shared-memory sub-communicators (matching ``MPI_Comm_split_type`` with ``MPI_COMM_TYPE_SHARED``):

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Communicator node_comm = comm.split_type(mpi::CommType::SHARED, comm.rank());
   // node_comm only contains processes on the same physical host / node
