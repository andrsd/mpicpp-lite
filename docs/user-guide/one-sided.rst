One-Sided Communication
=======================

One-sided communication (also known as Remote Memory Access, or RMA) decouples data transfer from process synchronization. Instead of both sender and receiver participating simultaneously, an origin process accesses the memory of a target process directly using a shared ``Window``.

.. index::
   single: Window
   single: RMA; remote memory access
   single: Window::create
   single: Window::allocate

Creating Windows
----------------

A ``Window`` exposes a segment of memory for remote reads and writes across a communicator.

Wrapping Existing Memory
^^^^^^^^^^^^^^^^^^^^^^^^

Expose an existing array or container using ``Window::create`` and ``std::span``:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <span>
   #include <vector>

   namespace mpi = mpicpp_lite;

   mpi::Communicator comm;

   std::vector<int> my_data(100, 0);
   mpi::Window win = mpi::Window::create(std::span { my_data }, mpi::Info {}, comm);

Allocating Remote Memory Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, allocate new memory and the window simultaneously using ``Window::allocate<T>``:

.. code-block:: cpp

   // Allocates memory for 100 integers on each process and creates the window
   auto [win, base_ptr] = mpi::Window::allocate<int>(100, mpi::Info {}, comm);

.. index::
   single: lock
   single: unlock
   single: passive target synchronization
   single: Window::get
   single: Window::put

Remote Memory Access Operations
-------------------------------

Reading Remote Memory
^^^^^^^^^^^^^^^^^^^^^

Use ``get()`` to copy data directly from a target process's window into local memory:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int target_rank = 0;
   int remote_offset = 10;
   int count = 5;
   std::vector<int> local_buffer(count);

   // Lock target window for shared read access
   win.lock(mpi::Lock::SHARED, target_rank);

   // Read 5 integers from target_rank starting at index 10
   win.get(local_buffer.data(), count, target_rank, remote_offset, count);

   win.unlock(target_rank);

Writing Remote Memory
^^^^^^^^^^^^^^^^^^^^^

Use ``put()`` to write local data directly into a target process's window:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int target_rank = 1;
   std::vector<int> payload = { 42, 43, 44 };

   // Lock target window for exclusive write access
   win.lock(mpi::Lock::EXCLUSIVE, target_rank);

   // Write payload into target_rank's window starting at index 0
   win.put(payload.data(), payload.size(), target_rank, 0, payload.size());

   win.unlock(target_rank);

.. index::
   single: fence
   single: active target synchronization

Synchronization with Fences
---------------------------

When all processes in the communicator access windows collectively in iterative phases, ``win.fence()`` provides active-target synchronization (acting like an epoch boundary):

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   // Start RMA access epoch
   win.fence();

   // Rank i writes data directly to neighbor rank i+1
   int neighbor = (comm.rank() + 1) % comm.size();
   win.put(&comm.rank(), 1, neighbor, 0, 1);

   // End RMA access epoch: guarantees all remote puts have finished
   win.fence();

Releasing Windows
-----------------

When remote access is no longer required, destroy the window resource with ``win.free()``:

.. code-block:: cpp

   win.free();
