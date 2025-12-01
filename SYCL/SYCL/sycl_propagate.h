#ifndef __SYCL_PROPAGATE
#define __SYCL_PROPAGATE

#include <sycl/sycl.hpp>
#include "sycl_propagate.h"
#include "sycl_defines.h"
#include "../map.h"
#include "../derivatives.h"

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void SYCL_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it,
			int offset,
	       float* pp, float* pc, float* qp, float* qc,
		   float* dev_ch1dxx, float* dev_ch1dyy, float* dev_ch1dzz,
		   float* dev_ch1dxy, float* dev_ch1dyz, float* dev_ch1dxz,
		   float* dev_v2px, float* dev_v2pz, float* dev_v2sz, float* dev_v2pn,
		   float* dev_pp, float* dev_pc,
		   float* dev_qp, float* dev_qc, sycl::queue q);

void SYCL_SwapArrays(float*& dev_pp, float*& dev_pc,
                     float*& dev_qp, float*& dev_qc);
#endif
