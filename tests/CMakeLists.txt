project(mpicpp-lite-test)

include(GoogleTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME}
    CartesianCommunicator_test.cpp
    Datatype_test.cpp
    Group_test.cpp
    MPI_test.cpp
    Window_test.cpp
    main.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        gtest_main
        gmock_main
        MPI::MPI_CXX
        mpicpp-lite
)

if (${fmt_FOUND})
    target_compile_definitions(${PROJECT_NAME} PUBLIC MPICPP_LITE_WITH_FMT)
    target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
endif()

target_code_coverage(${PROJECT_NAME})

target_sanitization(${PROJECT_NAME})

add_test(
    NAME mpicpp-lite-test-4
    COMMAND mpirun -np 4 --host localhost:4 $<TARGET_FILE:mpicpp-lite-test>
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

add_test(
    NAME mpicpp-lite-test-6
    COMMAND mpirun -np 6 --host localhost:6 $<TARGET_FILE:mpicpp-lite-test>
)
set_property(TEST mpicpp-lite-test-6 PROPERTY PROCESSORS 6)
set_property(TEST mpicpp-lite-test-6 PROPERTY PROCESSOR_AFFINITY TRUE)

if(MPICPP_LITE_CODE_COVERAGE)
    set_tests_properties(
        mpicpp-lite-test-6
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
