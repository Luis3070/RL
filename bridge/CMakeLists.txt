# ─────────────────────────────────────────────────────────────────────────────
# CMakeLists.txt — BHuman C++ Bridge for RL Integration
#
# This file is designed to be included into BHuman's existing CMake build.
#
# USAGE (two options):
#
#   Option A: Add to BHuman's main CMakeLists.txt (recommended)
#     1. Copy this file and bhuman_bridge.cpp into the BHuman tree:
#        cp bridge/bhuman_bridge.cpp  C:/Users/Alegu/Sabana2026/Src/Apps/RLBridge/
#        cp bridge/CMakeLists.txt     C:/Users/Alegu/Sabana2026/Make/CMake/RLBridge.cmake
#     2. Add to C:/Users/Alegu/Sabana2026/Make/CMake/CMakeLists.txt:
#        include(RLBridge.cmake)
#     3. Build: cd C:/Users/Alegu/Sabana2026/Make/Windows
#        cmake ../CMake -G "Visual Studio 17 2022"
#        cmake --build . --target RLBridge --config Release
#
#   Option B: Standalone build (requires BHuman already built)
#     1. Set BHUMAN_ROOT to point to BHuman source tree
#     2. cmake -B build -DBHUMAN_ROOT=C:/Users/Alegu/Sabana2026
#     3. cmake --build build --config Release
#
# OUTPUT: bhuman_bridge.exe (or RLBridge.exe depending on option)
# ─────────────────────────────────────────────────────────────────────────────

# ── Option A: As part of BHuman's CMake system (RLBridge.cmake) ──────────
# When included via BHuman's main CMakeLists.txt, all targets are available.
# This section uses BHuman's existing targets.

option(RLBRIDGE_ENABLE_ASAN "Build RLBridge with AddressSanitizer diagnostics" OFF)

if(DEFINED BHUMAN_PREFIX)
    # We are inside BHuman's CMake system
    set(BRIDGE_SRC "${BHUMAN_PREFIX}/Src/Apps/RLBridge/bhuman_bridge.cpp")

    if(EXISTS "${BRIDGE_SRC}")
        add_executable(RLBridge "${BRIDGE_SRC}")

        target_link_libraries(RLBridge PRIVATE
            B-Human${TARGET_SUFFIX}
            Framework${TARGET_SUFFIX}
            Platform${TARGET_SUFFIX}
            Streaming${TARGET_SUFFIX}
        )

        target_include_directories(RLBridge PRIVATE
            "${BHUMAN_PREFIX}/Src"
            "${BHUMAN_PREFIX}/Src/Libs"
        )

        if(BUILD_DESKTOP)
            target_link_libraries(RLBridge PRIVATE Flags::DebugInDevelop)
        endif()

        if(WINDOWS)
            target_link_libraries(RLBridge PRIVATE ws2_32 winmm)
            target_link_options(RLBridge PRIVATE /STACK:4194304)
            if(RLBRIDGE_ENABLE_ASAN AND MSVC)
                target_compile_options(RLBridge PRIVATE /fsanitize=address)
                target_link_options(RLBridge PRIVATE /fsanitize=address)
            endif()
        endif()

        target_compile_definitions(RLBridge PRIVATE
            CONFIGURATION=$<CONFIG>
        )

        if(WINDOWS)
            set(RLBRIDGE_ONNXRUNTIME_DLL
                "${BHUMAN_PREFIX}/Util/onnxruntime/lib/Windows/onnxruntime.dll")
            if(EXISTS "${RLBRIDGE_ONNXRUNTIME_DLL}")
                add_custom_command(TARGET RLBridge POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${RLBRIDGE_ONNXRUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:RLBridge>/onnxruntime.dll")
            endif()
        endif()

        set_property(TARGET RLBridge PROPERTY FOLDER "Apps")
        set_property(TARGET RLBridge PROPERTY
            RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_PREFIX}/Build/${PLATFORM}/RLBridge/$<CONFIG>")

        message(STATUS "RLBridge target added")
    else()
        message(WARNING "RLBridge source not found at ${BRIDGE_SRC}")
    endif()

    return()
endif()

# ── Option B: Standalone build ───────────────────────────────────────────
# For building outside BHuman's main CMake tree.
# Requires BHUMAN_ROOT and a pre-built BHuman library.

cmake_minimum_required(VERSION 3.17)
project(RLBridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# BHuman root directory
if(NOT DEFINED BHUMAN_ROOT)
    set(BHUMAN_ROOT "C:/Users/Alegu/Sabana2026" CACHE PATH "Path to BHuman source tree")
endif()

if(NOT EXISTS "${BHUMAN_ROOT}/Src")
    message(FATAL_ERROR "BHUMAN_ROOT (${BHUMAN_ROOT}) does not contain Src/. Set -DBHUMAN_ROOT=<path>")
endif()

message(STATUS "BHUMAN_ROOT = ${BHUMAN_ROOT}")

# Source
set(BRIDGE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/bhuman_bridge.cpp"
)

add_executable(bhuman_bridge ${BRIDGE_SOURCES})

# Include paths — BHuman's source structure
target_include_directories(bhuman_bridge PRIVATE
    "${BHUMAN_ROOT}/Src"
    "${BHUMAN_ROOT}/Src/Libs"
    "${BHUMAN_ROOT}/Src/Libs/Framework"
    "${BHUMAN_ROOT}/Src/Libs/Platform"
    "${BHUMAN_ROOT}/Src/Libs/Streaming"
    "${BHUMAN_ROOT}/Src/Libs/Math"
    "${BHUMAN_ROOT}/Src/Libs/MathBase"
    "${BHUMAN_ROOT}/Src/Libs/Debugging"
    "${BHUMAN_ROOT}/Src/Libs/Network"
    "${BHUMAN_ROOT}/Src/Libs/ImageProcessing"
    "${BHUMAN_ROOT}/Src/Libs/RobotParts"
)

# External dependencies
target_include_directories(bhuman_bridge SYSTEM PRIVATE
    "${BHUMAN_ROOT}/Util/Eigen"
    "${BHUMAN_ROOT}/Util/GameController/include"
    "${BHUMAN_ROOT}/Util/asmjit/src"
    "${BHUMAN_ROOT}/Util/CompiledNN/Src"
    "${BHUMAN_ROOT}/Util/CompiledNN/3rdParty/onnx"
)

target_link_libraries(bhuman_bridge PRIVATE
    Platform
    Streaming
)

# Platform defines
if(WIN32)
    target_compile_definitions(bhuman_bridge PRIVATE
        WINDOWS
        NOMINMAX
        _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
        _CRT_SECURE_NO_DEPRECATE
    )
    target_link_libraries(bhuman_bridge PRIVATE ws2_32 winmm)
    target_link_options(bhuman_bridge PRIVATE /STACK:4194304)

    if(MSVC)
        target_compile_options(bhuman_bridge PRIVATE
            /W3 /GF /GS- /EHsc /permissive- /Zc:preprocessor /Zc:__cplusplus
            /wd5105 /w34100 /w34189
        )
        if(RLBRIDGE_ENABLE_ASAN)
            target_compile_options(bhuman_bridge PRIVATE /fsanitize=address)
            target_link_options(bhuman_bridge PRIVATE /fsanitize=address)
        endif()
    endif()
endif()

target_compile_definitions(bhuman_bridge PRIVATE
    CONFIGURATION=$<CONFIG>
    __SSSE3__
)

# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON LINKING:
#
# The standalone build requires linking against BHuman's compiled object
# libraries. These are produced by BHuman's own CMake build system.
#
# Two approaches:
#
# 1. (Recommended) Build BHuman first, then link against its build artifacts:
#    - Build BHuman: cd C:/Users/Alegu/Sabana2026/Make/Windows && cmake --build . --config Develop
#    - Then find .obj/.lib files in Build/Windows/
#
# 2. (Advanced) Include BHuman's .cmake files to build everything together:
#    This is complex due to BHuman's many dependencies (FFTW, CompiledNN, etc.)
#
# For the initial iteration, Option A (integrating into BHuman's build) is
# strongly recommended. It avoids duplicating all dependency resolution.
# ─────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "══════════════════════════════════════════════════════════")
message(STATUS "  RLBridge - BHuman C++ Bridge for RL Integration")
message(STATUS "")
message(STATUS "  RECOMMENDED: Use Option A (integrate into BHuman build)")
message(STATUS "  See comments at top of this CMakeLists.txt for steps.")
message(STATUS "")
message(STATUS "  Quick start:")
message(STATUS "    1. mkdir -p ${BHUMAN_ROOT}/Src/Apps/RLBridge")
message(STATUS "    2. cp bhuman_bridge.cpp ${BHUMAN_ROOT}/Src/Apps/RLBridge/")
message(STATUS "    3. cp CMakeLists.txt ${BHUMAN_ROOT}/Make/CMake/RLBridge.cmake")
message(STATUS "    4. Add 'include(RLBridge.cmake)' to BHuman's CMakeLists.txt")
message(STATUS "    5. cmake --build . --target RLBridge --config Release")
message(STATUS "══════════════════════════════════════════════════════════")
message(STATUS "")
