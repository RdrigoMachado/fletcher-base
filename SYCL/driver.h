#ifndef __driver_h__
#define __driver_h__

#include "SYCL/sycl_defines.h"
#include <sycl/sycl.hpp>

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float* vpz, float* vsv, float* epsilon, float* delta,
                       float* phi, float* theta,

                       float* ch1dxx, float* ch1dyy, float* ch1dzz, float* ch1dxy,
                       float* ch1dyz, float* ch1dxz,
                       float* v2px, float* v2pz, float* v2sz, float* v2pn,

                       float*& dev_ch1dxx, float*& dev_ch1dyy, float*& dev_ch1dzz,
                       float*& dev_ch1dxy, float*& dev_ch1dyz, float*& dev_ch1dxz,
                       float*& dev_v2px, float*& dev_v2pz, float*& dev_v2sz, float*& dev_v2pn,
                       float*& dev_pp, float*& dev_pc, float*& dev_qp, float*& dev_qc, sycl::queue);

void DRIVER_Finalize();

void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
        const float dx, const float dy, const float dz, const float dt, const int it,
        int offset,
        float* pp, float* pc, float* qp, float* qc,
        float* dev_ch1dxx, float* dev_ch1dyy, float* dev_ch1dzz,
        float* dev_ch1dxy, float* dev_ch1dyz, float* dev_ch1dxz,
        float* dev_v2px, float* dev_v2pz, float* dev_v2sz, float* dev_v2pn,
        float* dev_pp, float* dev_pc,
        float* dev_qp, float* dev_qc, sycl::queue q);

void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float* pc, float* dev_pc, sycl::queue q);

void DRIVER_InsertSource(float dt, int it, int iSource, int offset, float* p, float* q, float src, sycl::queue);

#endif
