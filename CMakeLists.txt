cmake_minimum_required(VERSION VERSION 3.16)

project(mpicpp-lite
    VERSION 2.0.0
    LANGUAGES CXX
)

enable_language(C)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CodeCoverage)
include(Sanitization)

option(MPICPP_LITE_BUILD_TESTS "Build tests" NO)
option(MPICPP_LITE_WITH_FMT "Build with fmt support" NO)

if (MPICPP_LITE_WITH_FMT)
    find_package(fmt 11 REQUIRED)
endif()
find_package(MPI REQUIRED)

add_library(mpicpp-lite INTERFACE)
target_include_directories(mpicpp-lite
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mpicpp-lite INTERFACE MPI::MPI_C)
target_compile_features(mpicpp-lite
    INTERFACE
        cxx_std_20
)

# Install

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/mpicpp-lite-config.cmake.in
    mpicpp-lite-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpicpp-lite
    NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
    mpicpp-lite-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    TARGETS mpicpp-lite
    EXPORT MpiCppLiteTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT MpiCppLiteTargets
    FILE mpicpp-lite-targets.cmake
    NAMESPACE mpicpp-lite::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpicpp-lite
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mpicpp-lite-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mpicpp-lite-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpicpp-lite
)

# Tests

if (MPICPP_LITE_BUILD_TESTS)
    set(MPICPP_LITE_UNIT_TESTS_ROOT ${CMAKE_SOURCE_DIR}/tests CACHE PATH "" FORCE)
    mark_as_advanced(FORCE MPICPP_LITE_UNIT_TESTS_ROOT)

    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(docs)
