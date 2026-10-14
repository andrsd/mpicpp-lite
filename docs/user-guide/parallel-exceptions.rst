Parallel Exception Handling
============================

In distributed-memory MPI applications, handling C++ exceptions presents a major concurrency hazard: if a single rank encounters an error and throws an exception while other ranks continue normal execution, the healthy ranks inevitably deadlock at the next collective call (such as a barrier, broadcast, or reduction).

``mpicpp-lite`` provides a structured, deadlock-free solution through ``CollectiveExceptionHandler`` and ``ParallelException``.

.. index::
   single: exceptions; distributed
   single: CollectiveExceptionHandler
   single: ParallelException

The Distributed Exception Challenge
-----------------------------------

Consider what happens without collective handling:

.. code-block:: cpp

   // HAZARDOUS: Rank 1 aborts early, while ranks 0, 2, 3 wait at the reduction forever
   if (comm.rank() == 1) {
       throw std::runtime_error("File not found");
   }

   // DEADLOCK: Ranks 0, 2, and 3 hang here indefinitely
   comm.all_reduce(local_val, global_val, mpi::op::sum<>());

Using `CollectiveExceptionHandler`
----------------------------------

``CollectiveExceptionHandler`` coordinates exception states across all processes in a communicator:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <iostream>
   #include <stdexcept>

   namespace mpi = mpicpp_lite;

   int main(int argc, char * argv[])
   {
       mpi::Environment env(argc, argv);
       mpi::Communicator comm;

       mpi::CollectiveExceptionHandler handler(comm);

       try {
           // Simulate a localized failure on rank 1
           if (comm.rank() == 1) {
               throw std::runtime_error("Corrupted input data on rank 1");
           }

           // Checkpoint: verify all ranks are healthy before proceeding
           handler.sync();

           // Safe to execute collectives: all ranks reached this point cleanly
           double val = 1.0;
           comm.all_reduce(val, mpi::op::sum<>());
       }
       catch (std::exception & e) {
           // Synchronize exception across all ranks so none are left waiting
           auto par_exc = handler.sync(e);

           int failed_rank = par_exc.rank();

           // Broadcast the error message from the failing rank to all ranks
           par_exc.broadcast(comm);

           if (comm.rank() == 0) {
               std::cerr << "Collective failure originated on rank " << failed_rank 
                         << ": " << par_exc.what() << std::endl;
           }

           // All ranks are now in sync and can cleanly clean up or exit
           return 1;
       }

       return 0;
   }

How It Works
------------

1. **Inside the ``try`` block**: Calling ``handler.sync()`` performs an asynchronous non-blocking exchange across a dedicated private communicator duplicate. If any rank threw an exception, ``handler.sync()`` throws a ``ParallelException`` on the healthy ranks.
2. **Inside the ``catch`` block**: Calling ``handler.sync(e)`` exchanges the identity of the faulty rank.
3. **Broadcasting details**: Calling ``par_exc.broadcast(comm)`` sends the exception's ``what()`` message to all ranks, ensuring that diagnostic logs are accessible anywhere in the cluster.
