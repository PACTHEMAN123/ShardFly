cmake_minimum_required(VERSION 3.24)
project(ShardFly VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(SHARDFLY_BUILD_PYTHON "Build the optional PyTorch binding" OFF)
set(SHARDFLY_NCCL_ROOT "$ENV{SHARDFLY_NCCL_ROOT}" CACHE PATH "NCCL installation with Device API headers")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
find_package(CUDAToolkit REQUIRED)
find_path(NCCL_INCLUDE_DIR nccl.h HINTS "${SHARDFLY_NCCL_ROOT}/include" REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2
  HINTS "${SHARDFLY_NCCL_ROOT}/lib" "${SHARDFLY_NCCL_ROOT}/lib64" REQUIRED)
if(NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl_device.h")
  message(FATAL_ERROR "ShardFly requires NCCL Device API headers (NCCL 2.30.4+ for GIN)")
endif()
add_library(shardfly STATIC src/launch.cu src/runtime.cu)
add_library(ShardFly::shardfly ALIAS shardfly)
target_compile_features(shardfly PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(shardfly PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(shardfly PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  "${NCCL_INCLUDE_DIR}")
target_include_directories(shardfly PRIVATE include/shardfly)
target_link_libraries(shardfly PUBLIC CUDA::cudart "${NCCL_LIBRARY}" ${CMAKE_DL_LIBS})
target_compile_definitions(shardfly PUBLIC NCCL_OS_LINUX)
# Preserve the validated kernel compiler settings; do not change resource geometry.
target_compile_options(shardfly PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-extended-lambda;--expt-relaxed-constexpr;-Xptxas=-maxrregcount=96>)

if(SHARDFLY_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
  find_package(Torch REQUIRED)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
  Python_add_library(_native MODULE WITH_SOABI bindings/torch/bindings.cu)
  target_link_libraries(_native PRIVATE shardfly ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}")
  target_include_directories(_native PRIVATE include/shardfly ${TORCH_INCLUDE_DIRS})
  separate_arguments(TORCH_COMPILE_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  # The binding and library exchange std::string/vector values across their ABI.
  target_compile_options(shardfly PUBLIC ${TORCH_COMPILE_FLAGS})
  target_compile_options(_native PRIVATE ${TORCH_COMPILE_FLAGS}
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-extended-lambda;--expt-relaxed-constexpr;-Xptxas=-maxrregcount=96>)
  target_compile_features(_native PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(_native PRIVATE TORCH_EXTENSION_NAME=_native)
  set_target_properties(_native PROPERTIES INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${SHARDFLY_NCCL_ROOT}/lib;${SHARDFLY_NCCL_ROOT}/lib64")
  install(TARGETS _native LIBRARY DESTINATION shardfly)
endif()

install(TARGETS shardfly EXPORT ShardFlyTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/shardfly DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ShardFlyTargets NAMESPACE ShardFly:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ShardFly)
configure_package_config_file(cmake/ShardFlyConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/ShardFlyConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ShardFly)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ShardFlyConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ShardFlyConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ShardFlyConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ShardFly)
