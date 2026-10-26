nano_add_executable(MLGraphTests
        SOURCES
            GraphApplyTests.cpp
            GraphBlobTests.cpp
            GraphCoreMLTests.cpp
            GraphEncoderTests.cpp
            GraphPackageTests.cpp
            GraphProtobufTests.cpp
            GraphShapeTests.cpp
            GraphTextTests.cpp
        TARGETS eacp-ml-graph)

if (TARGET eacp-ml)
    target_link_libraries(MLGraphTests PRIVATE eacp-ml)
endif ()

# The suites generate and check millions of values, and build Whisper's
# encoder at its real sizes.
eacp_force_optimization(MLGraphTests)

if (TARGET eacp-ml-kernels)
    nano_add_executable(MLKernelTests
            SOURCES
                ActivationTests.cpp
                AttentionTests.cpp
                BandedAttentionTests.cpp
                LinearTests.cpp
                NormTests.cpp
                RoPETests.cpp
                SafetensorsFileTests.cpp
                SwiGLUTests.cpp
                TensorOpsTests.cpp
                TensorTests.cpp
            TARGETS eacp-ml-kernels)
endif ()

if (NOT EACP_HAS_COREML)
    return ()
endif ()

nano_add_executable(MLTests
        NO_MAIN
        SOURCES
            MultiArrayTests.cpp
            AsyncTests.cpp
            CacheTests.cpp
            PlacementTests.cpp
            ProgramTests.cpp
            TestMain.cpp
        TARGETS eacp-ml)

eacp_force_optimization(MLTests)
