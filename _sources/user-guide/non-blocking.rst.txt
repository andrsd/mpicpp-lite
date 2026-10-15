Non-Blocking Communication
==========================

Non-blocking communication decouples the initiation of a message transfer from its completion. This allows computation to overlap with data movement and prevents deadlock in cyclic communication patterns.

.. index::
   single: Communicator::isend
   single: Communicator::irecv
   single: isend
   single: irecv
   single: Request

Initiating Transfers
--------------------

Non-blocking operations begin with an ``i`` prefix (such as ``isend`` and ``irecv``) and return an ``mpi::Request`` handle immediately:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <vector>

   namespace mpi = mpicpp_lite;

   mpi::Communicator comm;

   if (comm.rank() == 0) {
       std::vector<double> send_buf = { 1.0, 2.0, 3.0, 4.0 };
       mpi::Request req = comm.isend(1, mpi::Tag(0), send_buf);

       // Perform local computation while message is in transit
       do_computation();

       // Wait until buffer can be safely reused or modified
       mpi::wait(req);
   }
   else if (comm.rank() == 1) {
       std::vector<double> recv_buf(4);
       mpi::Request req = comm.irecv(0, mpi::Tag(0), recv_buf);

       do_computation();

       mpi::wait(req);
   }

.. index::
   single: wait
   single: Status

Waiting for Completion
----------------------

To block until an operation has finished, use ``mpi::wait()``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Request req = comm.irecv(0, mpi::Tag(0), buffer);

   // Wait and receive the Status object:
   mpi::Status status = mpi::wait(req);

   std::cout << "Received " << status.count<double>() 
             << " elements from rank " << status.source() << std::endl;

.. index::
   single: wait_with_timeout

Waiting with a Timeout
^^^^^^^^^^^^^^^^^^^^^^

You can also specify a timeout in seconds to prevent indefinitely hanging processes:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   bool completed = mpi::wait_with_timeout(req, 2.5); // 2.5 seconds timeout
   if (!completed) {
       std::cerr << "Communication timed out!" << std::endl;
       req.cancel();
   }

.. index::
   single: test

Testing for Completion
----------------------

To check if a transfer has completed without blocking the calling thread, use ``mpi::test()``. It returns an ``std::optional<mpi::Status>``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Request req = comm.irecv(0, mpi::Tag(0), buffer);

   while (true) {
       auto status_opt = mpi::test(req);
       if (status_opt.has_value()) {
           // Request has completed
           std::cout << "Finished receiving from " << status_opt->source() << std::endl;
           break;
       }

       // Still in progress: perform background work
       do_background_work();
   }

.. index::
   single: wait_all
   single: wait_any
   single: wait_some
   single: test_all
   single: test_any

Managing Collections of Requests
--------------------------------

When coordinating transfers with multiple neighbors, you can collect ``mpi::Request`` handles in a ``std::vector<mpi::Request>``:

Waiting for All Requests
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   std::vector<mpi::Request> requests;

   // Post non-blocking sends to all neighbors
   for (int dest : neighbors) {
       requests.push_back(comm.isend(dest, mpi::Tag(1), data_out));
   }

   // Wait for all posted sends to complete:
   mpi::wait_all(requests);

Waiting for Any Request
^^^^^^^^^^^^^^^^^^^^^^^

``mpi::wait_any()`` blocks until at least one request completes, returning the index of the completed request:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::TestAnyResult result = mpi::wait_any(requests);
   std::cout << "Request at index " << result.index << " completed." << std::endl;

Testing Collections
^^^^^^^^^^^^^^^^^^^

You can also test whether all requests or any request in a collection has completed:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   if (mpi::test_all(requests)) {
       // All operations finished
   }

   auto any_result = mpi::test_any(requests);
   if (any_result.has_value()) {
       std::cout << "Request " << any_result->index << " finished." << std::endl;
   }

.. index::
   single: non-blocking collectives
   single: ibroadcast
   single: iall_reduce

Non-Blocking Collectives
------------------------

All standard collective operations have non-blocking counterparts returning an ``mpi::Request``:

- ``comm.ibroadcast(...)``
- ``comm.iscatter(...)``
- ``comm.igather(...)``
- ``comm.iall_gather(...)``
- ``comm.iall_to_all(...)``
- ``comm.ireduce(...)``
- ``comm.iall_reduce(...)``
- ``comm.iscan(...)``
- ``comm.iexscan(...)``

This enables overlapping heavy global reductions with local computation:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   double local_energy = compute_energy();
   double global_energy = 0.0;

   // Start global reduction asynchronously
   mpi::Request req = comm.iall_reduce(local_energy, global_energy, mpi::op::sum<>());

   // Compute boundary conditions or update local physics in parallel
   compute_local_physics();

   // Finish reduction before using the global result
   mpi::wait(req);
