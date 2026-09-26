cmake_minimum_required(VERSION 3.20)
project(room2 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# ---------------------------------------------------------------- dependencies
find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

# ---------------------------------------------------------------- shaders
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
     ${SHADER_SRC_DIR}/*.vert ${SHADER_SRC_DIR}/*.frag ${SHADER_SRC_DIR}/*.comp)
set(SHADER_OUTPUTS "")
foreach(SHADER ${SHADER_SOURCES})
    file(RELATIVE_PATH REL ${SHADER_SRC_DIR} ${SHADER})
    set(OUT ${SHADER_OUT_DIR}/${REL}.spv)
    add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 -I${SHADER_SRC_DIR} -O ${SHADER} -o ${OUT}
        DEPENDS ${SHADER} ${SHADER_SRC_DIR}/common.glsl
        COMMENT "glslc ${REL}"
        VERBATIM)
    list(APPEND SHADER_OUTPUTS ${OUT})
endforeach()
add_custom_target(room2_shaders ALL DEPENDS ${SHADER_OUTPUTS})

# ---------------------------------------------------------------- sources
# The tree is small and evolves quickly, so glob it: CONFIGURE_DEPENDS makes CMake
# re-run itself whenever a translation unit is added or removed.
file(GLOB_RECURSE ROOM2_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# While another contributor is mid-edit on a file, this lets the rest of the tree
# keep building against a second build directory.
option(ROOM2_EXCLUDE_IN_PROGRESS "Exclude sources that are being edited elsewhere" OFF)
if(ROOM2_EXCLUDE_IN_PROGRESS)
    list(FILTER ROOM2_SOURCES EXCLUDE REGEX "src/scene/weapon\.cpp")
    list(APPEND ROOM2_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/scene/weapon_stub.cpp)
endif()

add_executable(room2 ${ROOM2_SOURCES})
add_dependencies(room2 room2_shaders)

target_include_directories(room2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SDL2_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS})

target_link_libraries(room2 PRIVATE ${SDL2_LIBRARIES} Vulkan::Vulkan)
target_link_directories(room2 PRIVATE ${SDL2_LIBRARY_DIRS})
target_compile_options(room2 PRIVATE ${SDL2_CFLAGS_OTHER})

if(UNIX)
    target_link_libraries(room2 PRIVATE dl pthread)
endif()

# ---------------------------------------------------------------- tests
option(ROOM2_BUILD_TESTS "Build module unit tests" ON)
if(ROOM2_BUILD_TESTS)
    enable_testing()
    foreach(T physics audio mesh texture weapon player)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/${T}_test.cpp)
            add_executable(${T}_test tests/${T}_test.cpp)
            target_include_directories(${T}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${SDL2_INCLUDE_DIRS})
            target_link_directories(${T}_test PRIVATE ${SDL2_LIBRARY_DIRS})
            if(T STREQUAL "physics")
                target_sources(${T}_test PRIVATE src/physics/physics.cpp)
            elseif(T STREQUAL "player")
                target_sources(${T}_test PRIVATE src/game/player.cpp src/physics/physics.cpp
                                                src/core/log.cpp)
            elseif(T STREQUAL "weapon")
                target_sources(${T}_test PRIVATE src/scene/weapon.cpp src/scene/scene.cpp
                                                src/procgen/mesh.cpp src/core/log.cpp)
            elseif(T STREQUAL "mesh")
                target_sources(${T}_test PRIVATE src/procgen/mesh.cpp)
            elseif(T STREQUAL "texture")
                target_sources(${T}_test PRIVATE src/procgen/texture.cpp)
            else()
                target_sources(${T}_test PRIVATE src/audio/audio.cpp)
                target_link_libraries(${T}_test PRIVATE ${SDL2_LIBRARIES})
            endif()
            add_test(NAME ${T} COMMAND ${T}_test)
        endif()
    endforeach()
endif()
