Custom Datatypes
================

Standard MPI requires manually creating, committing, and passing ``MPI_Datatype`` handles whenever sending non-primitive C++ data structures.

In ``mpicpp-lite``, you register your data structure once by specializing ``mpi::DatatypeTraits<T>``. Once specialized, the library treats your custom type as a first-class citizen—allowing you to send, receive, broadcast, scatter, and gather instances or ``std::vector<T>`` collections without ever referencing an ``MPI_Datatype`` again.

.. index::
   single: DatatypeTraits
   single: type_create_struct
   single: structs; registering

Registering Custom Structs
--------------------------

To register a C++ struct, specialize ``mpi::DatatypeTraits<T>`` in the ``mpicpp_lite`` namespace and use ``mpi::type_create_struct()``:

.. code-block:: cpp

   #include "mpicpp-lite/mpicpp-lite.h"
   #include <cstddef>
   #include <vector>

   namespace mpi = mpicpp_lite;

   struct Particle {
       int id;
       double position[3];
       double mass;
       bool is_active;
   };

   // Register Particle with mpicpp-lite:
   namespace mpicpp_lite {

   template <>
   struct DatatypeTraits<Particle> {
       static MPI_Datatype
       get()
       {
           std::vector<MPI_Datatype> types = {
               MPI_INT,
               MPI_DOUBLE,
               MPI_DOUBLE,
               MPI_CXX_BOOL
           };

           std::vector<int> blk_lens = { 1, 3, 1, 1 };

           std::vector<MPI_Aint> offsets = {
               offsetof(Particle, id),
               offsetof(Particle, position),
               offsetof(Particle, mass),
               offsetof(Particle, is_active)
           };

           return type_create_struct(types, blk_lens, offsets);
       }
   };

   } // namespace mpicpp_lite

Sending and Receiving Custom Structs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once registered, communication is identical to standard primitive types:

.. code-block:: cpp

   mpi::Communicator comm;

   if (comm.rank() == 0) {
       Particle p { 101, { 0.1, 0.2, 0.3 }, 5.5, true };
       comm.send(1, mpi::Tag(0), p);
   }
   else if (comm.rank() == 1) {
       Particle p {};
       comm.recv(0, mpi::Tag(0), p);
       std::cout << "Particle ID: " << p.id 
                 << ", Mass: " << p.mass << std::endl;
   }

Broadcasting and Gathering Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collections of custom structs work automatically with ``std::vector<T>``:

.. code-block:: cpp

   std::vector<Particle> particles;
   if (comm.rank() == 0) {
       particles = {
           { 1, { 0.0, 1.0, 2.0 }, 1.0, true },
           { 2, { 3.0, 4.0, 5.0 }, 2.0, true }
       };
   }
   else {
       particles.resize(2);
   }

   comm.broadcast(particles, 0);

.. index::
   single: enums; registering
   single: type_contiguous

Registering Custom Enums
------------------------

Scoped enums (``enum class``) can be registered as contiguous blocks of their underlying integral type using ``mpi::type_contiguous()``:

.. code-block:: cpp

   enum class Color : unsigned char {
       RED,
       GREEN,
       BLUE
   };

   namespace mpicpp_lite {

   template <>
   struct DatatypeTraits<Color> {
       static MPI_Datatype
       get()
       {
           return type_contiguous(1, mpi_datatype<unsigned char>());
       }
   };

   } // namespace mpicpp_lite

Now you can send or reduce enums across processes:

.. code-block:: cpp

   Color chosen_color = Color::GREEN;
   comm.broadcast(chosen_color, 0);

.. index::
   single: fixed-size arrays
   single: templates; registering

Registering Templated Container Types
-------------------------------------

You can also specialize ``mpi::DatatypeTraits`` for templated math types, such as fixed-size vector or tensor structs:

.. code-block:: cpp

   template <typename T, int N>
   struct Vec {
       T data[N];
   };

   namespace mpicpp_lite {

   template <typename T, int N>
   struct DatatypeTraits<Vec<T, N>> {
       static MPI_Datatype
       get()
       {
           return type_contiguous(N, mpi_datatype<T>());
       }
   };

   } // namespace mpicpp_lite
