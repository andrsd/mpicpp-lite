Process Topologies
==================

Many scientific and numerical simulations map physical domains onto structured multi-dimensional grids of processes (e.g., 2D matrices or 3D volumes).

``mpicpp-lite`` provides ``CartesianCommunicator`` to represent Cartesian topologies, calculate coordinates, and query neighbors.

.. index::
   single: CartesianCommunicator
   single: topologies; Cartesian
   single: create_cartesian

Creating a Cartesian Grid
-------------------------

You create a Cartesian topology from any existing ``Communicator`` using ``create_cartesian()``:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <iostream>
   #include <vector>

   namespace mpi = mpicpp_lite;

   int main(int argc, char * argv[])
   {
       mpi::Environment env(argc, argv);
       mpi::Communicator comm;

       if (comm.size() != 6) {
           if (comm.rank() == 0)
               std::cerr << "Run with 6 processes (e.g., mpirun -np 6 ...)" << std::endl;
           return 1;
       }

       // Create a 2 x 3 process grid (2 rows, 3 columns)
       mpi::CartesianCommunicator cart_comm = comm.create_cartesian({ 2, 3 });

       std::cout << "Cartesian dimensions: " << cart_comm.dimensions() << std::endl;

       return 0;
   }

.. index::
   single: create_dims
   single: dimension decomposition

Automatic Dimension Factoring
-----------------------------

When you do not want to hardcode grid dimensions for arbitrary process counts, use ``mpi::create_dims()`` to factor the total number of processes into balanced Cartesian dimensions:

.. code-block:: cpp

   namespace mpi = mpicpp_lite;

   int num_dimensions = 2;
   std::vector<int> dims = mpi::create_dims(comm.size(), num_dimensions);

   mpi::CartesianCommunicator cart_comm = comm.create_cartesian(dims);

.. index::
   single: coordinates
   single: CartesianCommunicator::coords
   single: CartesianCommunicator::rank

Converting Between Ranks and Coordinates
----------------------------------------

``CartesianCommunicator`` provides bidirectional mapping between flat 1D MPI ranks and multi-dimensional grid coordinates:

Rank to Coordinates
^^^^^^^^^^^^^^^^^^^

Use ``coords(rank)`` to find the multi-dimensional grid position of a given process:

.. code-block:: cpp

   std::vector<int> my_coords = cart_comm.coords(cart_comm.rank());
   std::cout << "Rank " << cart_comm.rank() 
             << " has coordinates (" << my_coords[0] << ", " << my_coords[1] << ")" 
             << std::endl;

Coordinates to Rank
^^^^^^^^^^^^^^^^^^^

Use ``rank(...)`` with coordinate arguments to find the flat process rank at a specific grid position:

.. code-block:: cpp

   int target_rank = cart_comm.rank(1, 2); // Row 1, Column 2
   std::cout << "Process at (1, 2) has rank " << target_rank << std::endl;
