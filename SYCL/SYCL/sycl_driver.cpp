#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../driver.h"
#include "sycl_stuff.h"
#include "sycl_propagate.h"
#include "sycl_defines.h"
#include "sycl_insertsource.h"
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
                       float*& dev_pp, float*& dev_pc, float*& dev_qp, float*& dev_qc, sycl::queue q)
{
    int msize_vol = sx*sy*sz;
    ch1dxx = sycl::malloc_host<float>(msize_vol, q);
    ch1dyy = sycl::malloc_host<float>(msize_vol, q);
    ch1dzz = sycl::malloc_host<float>(msize_vol, q);
    ch1dxy = sycl::malloc_host<float>(msize_vol, q);
    ch1dyz = sycl::malloc_host<float>(msize_vol, q);
    ch1dxz = sycl::malloc_host<float>(msize_vol, q);
    for (int i=0; i<msize_vol; i++) {
        float sinTheta=sin(theta[i]);
        float cosTheta=cos(theta[i]);
        float sin2Theta=sin(2.0*theta[i]);
        float sinPhi=sin(phi[i]);
        float cosPhi=cos(phi[i]);
        float sin2Phi=sin(2.0*phi[i]);
        ch1dxx[i]=sinTheta*sinTheta * cosPhi*cosPhi;
        ch1dyy[i]=sinTheta*sinTheta * sinPhi*sinPhi;
        ch1dzz[i]=cosTheta*cosTheta;
        ch1dxy[i]=sinTheta*sinTheta * sin2Phi;
        ch1dyz[i]=sin2Theta         * sinPhi;
        ch1dxz[i]=sin2Theta         * cosPhi;
    }
#ifdef _DUMP
{
const int iPrint = ind(bord+1,bord+1,bord+1);
printf("ch1dxx=%f; ch1dyy=%f; ch1dzz=%f; ch1dxy=%f; ch1dxz=%f; ch1dyz=%f\n",
ch1dxx[iPrint], ch1dyy[iPrint], ch1dzz[iPrint], ch1dxy[iPrint], ch1dxz[iPrint], ch1dyz[iPrint]);
}
#endif

    // coeficients of H1 and H2 at PDEs

    v2px = sycl::malloc_host<float>(msize_vol, q);
    v2pz = sycl::malloc_host<float>(msize_vol, q);
    v2sz = sycl::malloc_host<float>(msize_vol, q);
    v2pn = sycl::malloc_host<float>(msize_vol, q);
    for (int i=0; i<msize_vol; i++){
        v2sz[i]=vsv[i]*vsv[i];
        v2pz[i]=vpz[i]*vpz[i];
        v2px[i]=v2pz[i]*(1.0+2.0*epsilon[i]);
        v2pn[i]=v2pz[i]*(1.0+2.0*delta[i]);
    }

#ifdef _DUMP
{
const int iPrint=ind(bord+1,bord+1,bord+1);
printf("vsv=%e; vpz=%e, v2pz=%e\n",
     vsv[iPrint], vpz[iPrint], v2pz[iPrint]);
printf("v2sz=%e; v2pz=%e, v2px=%e, v2pn=%e\n",
     v2sz[iPrint], v2pz[iPrint], v2px[iPrint], v2pn[iPrint]);
}
#endif

	SYCL_Initialize(sx, sy, sz, bord,
	       ch1dxx, ch1dyy, ch1dzz,
	       ch1dxy, ch1dyz, ch1dxz,
	       v2px, v2pz, v2sz, v2pn,
		   dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
		   dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
		   dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
		   dev_pp, dev_pc,
		   dev_qp, dev_qc, q);
}



void DRIVER_Finalize()
{
	SYCL_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float* pc, float* dev_pc, sycl::queue q)
{
	SYCL_Update_pointers(sx, sy, sz, pc, dev_pc, q);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
                      int offset,
                    float* pp, float* pc, float* qp, float* qc,
                    float* dev_ch1dxx, float* dev_ch1dyy, float* dev_ch1dzz,
                    float* dev_ch1dxy, float* dev_ch1dyz, float* dev_ch1dxz,
                    float* dev_v2px, float* dev_v2pz, float* dev_v2sz, float* dev_v2pn,
                    float* dev_pp, float* dev_pc,
                    float* dev_qp, float* dev_qc, sycl::queue q)
{

	// SYCL_Propagate also does TimeForward
	   SYCL_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it,
						offset,
	                    pp,    pc,    qp,    qc,
						dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
						dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
						dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
						dev_pp, dev_pc,
						dev_qp, dev_qc, q);
}


void DRIVER_InsertSource(float dt, int it, int iSource, int offset, float* dev_pc, float* dev_qc, float src, sycl::queue q)
{
	SYCL_InsertSource(src, iSource, offset, dev_pc, dev_qc, q);
}
