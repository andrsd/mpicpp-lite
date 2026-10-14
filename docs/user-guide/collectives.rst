Collective Operations
=====================

Collective operations involve all processes within a communicator. In ``mpicpp-lite``, collectives are type-safe and automatically determine buffer sizes when containers like ``std::vector<T>`` are used.

.. index::
   single: Communicator::barrier
   single: barrier; synchronization

Barrier Synchronization
-----------------------

To block all processes until every process in the communicator reaches the call:

.. code-block:: cpp

   comm.barrier();

.. index::
   single: Communicator::broadcast
   single: broadcast

Broadcasting
------------

Broadcasts a scalar or vector from a root process to all other processes:

.. code-block:: cpp

   // Broadcast a scalar
   int config_param = (comm.rank() == 0) ? 42 : 0;
   comm.broadcast(config_param, 0); // Root is 0

   // Broadcast a vector
   std::vector<double> coefficients;
   if (comm.rank() == 0)
       coefficients = { 1.5, 2.7, 3.9 };
   else
       coefficients.resize(3);

   comm.broadcast(coefficients, 0);

.. index::
   single: Communicator::scatter
   single: Communicator::gather
   single: Communicator::all_gather
   single: scatter
   single: gather
   single: all_gather

Data Distribution
-----------------

Scatter
^^^^^^^

Divides a buffer on the root process into equal segments and sends one segment to each process:

.. code-block:: cpp

   int chunk_size = 10;
   std::vector<int> send_buf;
   if (comm.rank() == 0)
       send_buf.resize(comm.size() * chunk_size, 99);

   std::vector<int> recv_chunk(chunk_size);
   comm.scatter(send_buf, recv_chunk, 0);

Gather
^^^^^^

Collects an equal-sized segment from each process onto the root process:

.. code-block:: cpp

   std::vector<int> local_chunk(chunk_size, comm.rank());
   std::vector<int> gathered;
   if (comm.rank() == 0)
       gathered.resize(comm.size() * chunk_size);

   comm.gather(local_chunk, gathered, 0);

AllGather
^^^^^^^^^

Collects data from all processes and distributes the combined result to all processes:

.. code-block:: cpp

   std::vector<int> local_data = { comm.rank() };
   std::vector<int> all_data(comm.size());

   comm.all_gather(local_data, all_data);

.. index::
   single: Communicator::all_to_all
   single: all_to_all

All-to-All
----------

Transposes data across all processes such that the ``i``-th segment of each process is sent to process ``i``:

.. code-block:: cpp

   std::vector<int> send_data(comm.size());
   for (int i = 0; i < comm.size(); ++i)
       send_data[i] = comm.rank() * 100 + i;

   std::vector<int> recv_data(comm.size());
   comm.all_to_all(send_data, recv_data);

.. index::
   single: Communicator::reduce
   single: Communicator::all_reduce
   single: reduce
   single: all_reduce

Reductions
----------

Reductions combine elements across processes using an associative binary operator (such as summation or maximum).

Rooted Reduction
^^^^^^^^^^^^^^^^

The result is only collected on the specified root rank:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   double local_val = 1.0;
   double total_sum = 0.0;
   comm.reduce(local_val, total_sum, mpi::op::sum<>(), 0);

   if (comm.rank() == 0)
       std::cout << "Sum: " << total_sum << std::endl;

All-Processes Reduction
^^^^^^^^^^^^^^^^^^^^^^^

Computes the reduction and delivers the result to every process:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   double local_max = compute_local_max();
   double global_max;
   comm.all_reduce(local_max, global_max, mpi::op::max<>());

In-Place Reductions
^^^^^^^^^^^^^^^^^^^

If you do not need to preserve the input value, perform the reduction in place without allocating a separate output buffer:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   double val = 3.5;
   comm.all_reduce(val, mpi::op::sum<>());
   // `val` now holds the global sum

.. index::
   single: Communicator::scan
   single: Communicator::exscan
   single: scan
   single: exscan

Prefix Scans
------------

Inclusive Scan
^^^^^^^^^^^^^^

Computes an inclusive prefix reduction (process :math:`k` receives the reduction from ranks :math:`0 \dots k`):

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int local_items = comm.rank() + 1;
   int offset;
   comm.scan(local_items, offset, mpi::op::sum<>());

Exclusive Scan
^^^^^^^^^^^^^^

Computes an exclusive prefix reduction (process :math:`k` receives the reduction from ranks :math:`0 \dots k-1`):

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int local_items = 10;
   int start_index = 0;
   comm.exscan(local_items, start_index, mpi::op::sum<>());

.. index::
   single: operations
   single: reduction operations

Predefined Reduction Operations
-------------------------------

``mpicpp-lite`` provides functors under the ``mpi::op`` namespace corresponding to standard MPI reduction operators:

Arithmetic Operations
^^^^^^^^^^^^^^^^^^^^^
- ``mpi::op::sum<>``: Summation (``MPI_SUM``)
- ``mpi::op::prod<>``: Product (``MPI_PROD``)
- ``mpi::op::min<>``: Minimum value (``MPI_MIN``)
- ``mpi::op::max<>``: Maximum value (``MPI_MAX``)

Logical Operations
^^^^^^^^^^^^^^^^^^
- ``mpi::op::land<>`` (or ``mpi::op::logical_and<>``): Logical AND (``MPI_LAND``)
- ``mpi::op::lor<>`` (or ``mpi::op::logical_or<>``): Logical OR (``MPI_LOR``)
- ``mpi::op::lxor<>`` (or ``mpi::op::logical_xor<>``): Logical XOR (``MPI_LXOR``)

Bitwise Operations
^^^^^^^^^^^^^^^^^^
- ``mpi::op::band<>`` (or ``mpi::op::bitwise_and<>``): Bitwise AND (``MPI_BAND``)
- ``mpi::op::bor<>`` (or ``mpi::op::bitwise_or<>``): Bitwise OR (``MPI_BOR``)
- ``mpi::op::bxor<>`` (or ``mpi::op::bitwise_xor<>``): Bitwise XOR (``MPI_BXOR``)

.. index::
   single: minloc
   single: maxloc
   single: ValueLoc

Extremum with Location
^^^^^^^^^^^^^^^^^^^^^^
Finds the global extremum and the process rank where it occurred using ``mpi::ValueLoc<T>``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::ValueLoc<double> local { local_error, comm.rank() };
   mpi::ValueLoc<double> global;

   comm.all_reduce(local, global, mpi::op::minloc<>());

   std::cout << "Min error: " << global.value 
             << " from rank: " << global.loc << std::endl;
