cmake_minimum_required(VERSION 3.22)
project(ciir_vr_simulation
    VERSION 0.1.0
    DESCRIPTION "Immersive VR CIIR → QuASIM → QRATUM Simulation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Options
# ============================================================
option(ENABLE_VR        "Enable OpenXR VR support"         OFF)
option(ENABLE_IMGUI     "Enable Dear ImGui dashboards"     ON)
option(ENABLE_VALIDATION "Enable Vulkan validation layers" ON)

# ============================================================
# Find packages
# ============================================================
find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 QUIET)
find_package(Threads REQUIRED)

if(ENABLE_VR)
    find_package(OpenXR QUIET)
    if(NOT OpenXR_FOUND)
        message(WARNING "OpenXR not found — VR disabled")
        set(ENABLE_VR OFF)
    endif()
endif()

# ============================================================
# Shader compilation (GLSL → SPIR-V)
# ============================================================
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")

set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SPIRV_DIR  "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_DIR})

file(GLOB SHADER_SOURCES
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
    "${SHADER_DIR}/*.comp"
)

set(SPIRV_OUTPUTS "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUT "${SPIRV_DIR}/${SHADER_NAME}.spv")
    if(GLSLC)
        add_custom_command(
            OUTPUT  ${SPIRV_OUT}
            COMMAND ${GLSLC} -O ${SHADER} -o ${SPIRV_OUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
    else()
        # Fallback: just copy the source so the build doesn't fail
        add_custom_command(
            OUTPUT  ${SPIRV_OUT}
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER} ${SPIRV_OUT}
            DEPENDS ${SHADER}
            COMMENT "Copying shader ${SHADER_NAME} (glslc not found)"
        )
    endif()
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUT})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# ============================================================
# Sources
# ============================================================
set(CIIR_SOURCES
    src/ciir/ciir_state.cpp
    src/ciir/ciir_observer.cpp
    src/ciir/ciir_visualizer.cpp
)

set(QUASIM_SOURCES
    src/quasim/quasim_tensor.cpp
    src/quasim/tensor_runtime.cpp
    src/quasim/tensor_visualizer.cpp
)

set(QRATUM_SOURCES
    src/qratum/qratum_hardware.cpp
    src/qratum/execution_scheduler.cpp
    src/qratum/performance_dashboard.cpp
)

set(RENDERER_SOURCES
    src/renderer/vulkan_context.cpp
    src/renderer/vr_context.cpp
    src/renderer/mesh.cpp
    src/renderer/camera.cpp
)

set(ALL_SOURCES
    src/main.cpp
    src/simulation_engine.cpp
    ${CIIR_SOURCES}
    ${QUASIM_SOURCES}
    ${QRATUM_SOURCES}
    ${RENDERER_SOURCES}
)

# ============================================================
# Executable
# ============================================================
add_executable(ciir_vr_sim ${ALL_SOURCES})

target_include_directories(ciir_vr_sim PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(ciir_vr_sim PRIVATE
    Vulkan::Vulkan
    Threads::Threads
)

if(glfw3_FOUND)
    target_link_libraries(ciir_vr_sim PRIVATE glfw)
    target_compile_definitions(ciir_vr_sim PRIVATE HAS_GLFW=1)
endif()

if(ENABLE_VR AND OpenXR_FOUND)
    target_link_libraries(ciir_vr_sim PRIVATE OpenXR::openxr_loader)
    target_compile_definitions(ciir_vr_sim PRIVATE ENABLE_VR=1)
endif()

if(ENABLE_VALIDATION)
    target_compile_definitions(ciir_vr_sim PRIVATE ENABLE_VULKAN_VALIDATION=1)
endif()

add_dependencies(ciir_vr_sim compile_shaders)

# Copy shader directory path as define
target_compile_definitions(ciir_vr_sim PRIVATE
    SHADER_DIR="${SPIRV_DIR}"
)

# ============================================================
# Install
# ============================================================
install(TARGETS ciir_vr_sim RUNTIME DESTINATION bin)
install(DIRECTORY ${SPIRV_DIR}/ DESTINATION share/ciir_vr_sim/shaders)
