#ifndef __KOKKOS_DEFINES
#define __KOKKOS_DEFINES

#include <Kokkos_Core_fwd.hpp>
#define restrict __restrict__
#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X+2*NPOP)
#define TOTAL_Y (BSIZE_Y+2*NPOP)


#include <stdio.h>
#include <Kokkos_Core.hpp>

using HostMemSpace   = Kokkos::HostSpace;

#ifdef KOKKOS_ENABLE_CUDA
  #include <cuda_runtime.h>
  using ExecutionSpace = Kokkos::Cuda;
  using DeviceMemSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
  #include <hip/hip_runtime.h>
  using ExecutionSpace = Kokkos::HIP;
  using DeviceMemSpace = Kokkos::HIPSpace;
#else
  #error "Either KOKKOS_ENABLE_CUDA or KOKKOS_ENABLE_HIP must be defined"
#endif

using DeviceViewFloat1D = Kokkos::View<float*, DeviceMemSpace>;
using DeviceViewFloat2D = Kokkos::View<float**, DeviceMemSpace>;
using DeviceViewFloat3D = Kokkos::View<float***, DeviceMemSpace>;
using HostViewFloat1D = Kokkos::View<float*, HostMemSpace>;
using HostViewFloat2D = Kokkos::View<float**, HostMemSpace>;
using HostViewFloat3D = Kokkos::View<float***, HostMemSpace>;

#define KOKKOS_CALL(call) do{      \
   const KOKKOSError_t err=call;         \
   if (err != KOKKOSSuccess)       \
   {                             \
     fprintf(stderr, "KOKKOS ERROR: %s on %s:%d\n", KOKKOSGetErrorString(err), __FILE__, __LINE__);\
     exit(1);                    \
   }}while(0)

#endif
