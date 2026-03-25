# Changelog

All notable changes to AgierCompute will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Add support for more kernel operations (convolution, pooling)
- Implement async kernel execution with streams
- Add memory pool for better performance
- Support for Metal backend
- Add Python bindings

## [1.0.0] - 2026-03-25

### Added

#### Core Features
- Cross-platform GPU computing framework with unified API
- Support for 4 backends: CPU, CUDA, OpenCL, Vulkan
- Generic kernel definition with type placeholder `T`
- Built-in variables: `GID`, `LID`, `BSIZE`, `BARRIER()`

#### Device Management
- `ace_device_count()` - Get number of devices by type
- `ace_device_get()` - Get device handle
- `ace_device_get_all()` - Get all available devices
- `ace_device_select_best()` - Auto-select best device (GPU preferred)
- `ace_device_props()` - Get device properties
- `ace_device_print_info()` - Print device information

#### Memory Management
- `ace_buffer_alloc()` - Allocate device memory
- `ace_buffer_free()` - Free device memory
- `ace_buffer_write()` - Write data to device (async)
- `ace_buffer_read()` - Read data from device (auto-sync)

#### Multi-Device Data Parallel
- `ace_buffer_alloc_sharded()` - Allocate sharded buffers across devices
- `ace_buffer_free_sharded()` - Free sharded buffers
- `ace_buffer_write_sharded()` - Write to sharded buffers
- `ace_buffer_read_sharded()` - Read from sharded buffers
- `ace_kernel_invoke_sharded()` - Execute kernel across multiple devices
- `ace_finish_all()` - Wait for all devices to complete

#### Kernel Execution
- `ace_kernel_invoke()` - Execute kernel with 1D scheduling
- `ace_kernel_launch()` - Execute kernel with custom 3D scheduling
- `ace_register_kernel()` - Register custom kernel

#### Pre-built Kernels
- Vector operations: `vec_add`, `vec_sub`, `vec_mul`
- Activation functions: `relu`, `sigmoid`, `tanh`, `softmax`
- Math functions: `exp`, `log`, `sqrt`, `square`, `abs`
- Data operations: `scale`, `fill`, `copy`, `negate`
- Linear algebra: `gemm` (matrix multiplication), `dot` (dot product)

#### Data Types
- `ACE_DTYPE_FLOAT32` - 32-bit float
- `ACE_DTYPE_FLOAT64` - 64-bit double
- `ACE_DTYPE_INT32` - 32-bit int
- `ACE_DTYPE_INT64` - 64-bit long

#### Error Handling
- Comprehensive error codes: `ACE_OK`, `ACE_ERROR`, `ACE_ERROR_MEM`, etc.
- `ace_error_string()` - Get error description

#### Documentation
- README.md - Project overview and quick start guide
- docs/API.md - Complete API reference
- examples/ - Multiple example programs

#### Examples
- `simple_test` - Basic CPU backend test
- `ace_demo` - Multi-backend demo
- `user_kernels` - User-defined kernel examples
- `multi_device_test` - Multi-device parallel execution tests
- `unit_tests` - Complete unit test suite (24 tests)
- `benchmark` - Performance benchmark suite

### Backend Details

#### CPU Backend
- Multi-threaded execution with thread pool
- Automatic CPU core detection
- Optimized implementations for all pre-built kernels
- Support for generic kernel execution

#### CUDA Backend
- NVRTC runtime compilation
- Support for all CUDA-capable GPUs
- Automatic compute capability detection

#### OpenCL Backend
- Cross-platform GPU/CPU support
- OpenCL 1.2+ compatibility
- Automatic platform and device selection

#### Vulkan Backend
- Modern graphics API compute shaders
- SPIR-V compilation via shaderc
- Support for discrete and integrated GPUs

### Changed
- None (initial release)

### Deprecated
- None

### Removed
- None

### Fixed
- None (initial release)

### Security
- None

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-03-25 | Initial release with CPU/CUDA/OpenCL/Vulkan support |
