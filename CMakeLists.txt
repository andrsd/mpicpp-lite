cmake_minimum_required(VERSION VERSION 3.16)

project(mpicpp-lite
    VERSION 1.0.0
    LANGUAGES CXX
)

enable_language(C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(FetchContent)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Sanitization.cmake)

option(MPICPP_LITE_BUILD_TESTS "Build tests" NO)

find_package(fmt 8.0 QUIET)
find_package(MPI REQUIRED)

if (MPICPP_LITE_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.12.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    mark_as_advanced(FORCE
        BUILD_GMOCK
        INSTALL_GTEST
        FETCHCONTENT_SOURCE_DIR_GOOGLETEST
        FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST
    )

    FetchContent_MakeAvailable(googletest)
    mark_as_advanced(FORCE
        FETCHCONTENT_BASE_DIR
        FETCHCONTENT_FULLY_DISCONNECTED
        FETCHCONTENT_QUIET
        FETCHCONTENT_UPDATES_DISCONNECTED
    )
endif ()

# Install

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpicpp-lite
    FILES_MATCHING PATTERN "*.h"
)

# Tests

if (MPICPP_LITE_BUILD_TESTS)
    set(MPICPP_LITE_UNIT_TESTS_ROOT ${CMAKE_SOURCE_DIR}/tests CACHE PATH "" FORCE)
    mark_as_advanced(FORCE MPICPP_LITE_UNIT_TESTS_ROOT)

    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(docs)
