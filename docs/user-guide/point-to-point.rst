Point-to-Point Communication
============================

Point-to-point communication transfers data directly between two processes using a source rank, a destination rank, and an identifying message tag.

.. index::
   single: Tag

Message Tags
------------

``mpicpp-lite`` uses the strongly-typed ``mpi::Tag`` class rather than raw integers to prevent accidental argument transposition. You construct tags explicitly with ``mpi::Tag(value)``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Tag default_tag(0);
   mpi::Tag data_tag(100);

.. index::
   single: Communicator::send
   single: Communicator::recv
   single: send
   single: recv

Sending and Receiving Scalars
-----------------------------

Sending a single value (such as an ``int`` or ``double``) does not require manual buffer allocation or datatype specification:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Communicator comm;

   if (comm.rank() == 0) {
       double value = 42.195;
       comm.send(1, mpi::Tag(0), value); // dest=1, tag=0, value
   }
   else if (comm.rank() == 1) {
       double value = 0.0;
       comm.recv(0, mpi::Tag(0), value); // source=0, tag=0, out-value
       std::cout << "Received: " << value << std::endl;
   }

.. index::
   single: std::vector; sending and receiving

Sending and Receiving Containers
--------------------------------

``mpicpp-lite`` provides first-class support for ``std::vector<T>``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   if (comm.rank() == 0) {
       std::vector<int> numbers = { 10, 20, 30, 40, 50 };
       comm.send(1, mpi::Tag(100), numbers);
   }
   else if (comm.rank() == 1) {
       std::vector<int> numbers(5);
       comm.recv(0, mpi::Tag(100), numbers);
       // numbers now contains { 10, 20, 30, 40, 50 }
   }

Sending Raw Pointers and Array Slices
-------------------------------------

When working with pre-allocated memory or continuous arrays, you can pass a pointer and length:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   float buffer[256];
   // Send 128 elements starting at buffer
   comm.send(1, mpi::Tag(0), 128, buffer);

   // Receive 128 elements into buffer
   comm.recv(0, mpi::Tag(0), 128, buffer);

.. index::
   single: Status

Inspecting Message Metadata
---------------------------

You can query metadata about received messages (such as the actual sender, tag, or number of elements received) by passing a ``Status`` object:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   mpi::Status status;
   std::vector<double> data(100);

   // Accept message from any source with any tag:
   comm.recv(mpi::ANY_SOURCE, mpi::ANY_TAG, data, status);

   std::cout << "Sender: " << status.source() << "\n"
             << "Tag:    " << status.tag().value() << "\n"
             << "Count:  " << status.count<double>() << std::endl;

.. index::
   single: Communicator::sendrecv
   single: sendrecv

Simultaneous Send and Receive
-----------------------------

To avoid circular deadlocks when exchanging data between pairs of processes, use ``sendrecv()``:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int next_rank = (comm.rank() + 1) % comm.size();
   int prev_rank = (comm.rank() - 1 + comm.size()) % comm.size();

   int out_msg = comm.rank();
   int in_msg = 0;

   // Send to next rank, receive from previous rank
   comm.sendrecv(next_rank, mpi::Tag(0), out_msg,
                 prev_rank, mpi::Tag(0), in_msg);
