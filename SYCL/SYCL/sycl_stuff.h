#ifndef _SYCL_STUFF
#define _SYCL_STUFF

#include <sycl/sycl.hpp>
#include "sycl_defines.h"

void SYCL_Initialize(
    const int sx, const int sy, const int sz, const int bord,
    float* ch1dxx, float* ch1dyy, float* ch1dzz,
    float* ch1dxy, float* ch1dyz, float* ch1dxz,
    float* v2px, float* v2pz, float* v2sz, float* v2pn,
    float*& dev_ch1dxx, float*& dev_ch1dyy, float*& dev_ch1dzz,
    float*& dev_ch1dxy, float*& dev_ch1dyz, float*& dev_ch1dxz,
    float*& dev_v2px, float*& dev_v2pz, float*& dev_v2sz, float*& dev_v2pn,
    float*& dev_pp, float*& dev_pc,
    float*& dev_qp, float*& dev_qc,
    sycl::queue q);


void SYCL_Finalize();

void SYCL_Update_pointers(const int sx, const int sy, const int sz, float* pc, float* dev_pc, sycl::queue q);

#endif
