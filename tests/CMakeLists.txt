project(mpicpp-lite-test)

include(GoogleTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME}
    Datatype_test.cpp
    Group_test.cpp
    MPI_test.cpp
    main.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/mpicpp-lite>
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    gtest_main
    gmock_main
    MPI::MPI_CXX
    fmt::fmt
)

target_code_coverage(${PROJECT_NAME})

target_sanitization(${PROJECT_NAME})

add_test(
    NAME mpicpp-lite-test-4
    COMMAND mpirun -np 4 $<TARGET_FILE:mpicpp-lite-test>
)
set_property(TEST mpicpp-lite-test-4 PROPERTY PROCESSORS 4)
set_property(TEST mpicpp-lite-test-4 PROPERTY PROCESSOR_AFFINITY TRUE)

if(MPICPP_LITE_CODE_COVERAGE)
    set_tests_properties(
        mpicpp-lite-test-4
        PROPERTIES
        ENVIRONMENT LLVM_PROFILE_FILE=mpicpp-lite-test-%4m.profraw
    )
endif()

# single-threaded
add_test(
    NAME mpicpp-lite-test-1
    COMMAND mpicpp-lite-test
)

if(MPICPP_LITE_CODE_COVERAGE)
    set_tests_properties(
        mpicpp-lite-test-1
        PROPERTIES
        ENVIRONMENT LLVM_PROFILE_FILE=mpicpp-lite-test-1.profraw
    )
endif()
