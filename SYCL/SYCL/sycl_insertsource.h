#ifndef __SYCL_SOURCE
#define __SYCL_SOURCE

#include <sycl/sycl.hpp>
#include "sycl_defines.h"

void SYCL_InsertSource(const float val, const int iSource, int offset, float* dev_pc, float* dev_qc, sycl::queue q);

#endif
