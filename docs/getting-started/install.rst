Installation & Setup
====================

``mpicpp-lite`` is a header-only library requiring a C++20 compliant compiler and an MPI implementation (such as OpenMPI or MPICH).

Using CMake with ``find_package``
---------------------------------

If ``mpicpp-lite`` is installed on your system (or in a conda or spack environment), CMake automatically resolves the underlying MPI dependency transitively:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.16)
   project(my_mpi_app LANGUAGES CXX)

   find_package(mpicpp-lite REQUIRED)

   add_executable(my_mpi_app main.cpp)
   target_link_libraries(my_mpi_app PRIVATE mpicpp-lite::mpicpp-lite)

Using CMake with ``FetchContent``
---------------------------------

You can also integrate ``mpicpp-lite`` directly into your CMake project without pre-installing it:

.. code-block:: cmake

   include(FetchContent)

   FetchContent_Declare(
       mpicpp_lite
       GIT_REPOSITORY https://github.com/andrsd/mpicpp-lite.git
       GIT_TAG        main
   )
   FetchContent_MakeAvailable(mpicpp_lite)

   target_link_libraries(my_mpi_app PRIVATE mpicpp-lite)
