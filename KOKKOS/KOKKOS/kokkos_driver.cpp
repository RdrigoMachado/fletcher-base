#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"../driver.h"
#include "kokkos_defines.h"
#include"kokkos_stuff.h"
#include"kokkos_propagate.h"
#include"kokkos_insertsource.h"

// Global device vars
// float* dev_ch1dxx=NULL;
// float* dev_ch1dyy=NULL;
// float* dev_ch1dzz=NULL;
// float* dev_ch1dxy=NULL;
// float* dev_ch1dyz=NULL;
// float* dev_ch1dxz=NULL;
// float* dev_v2px=NULL;
// float* dev_v2pz=NULL;
// float* dev_v2sz=NULL;
// float* dev_v2pn=NULL;
// float* dev_pp=NULL;
// float* dev_pc=NULL;
// float* dev_qp=NULL;
// float* dev_qc=NULL;


// #define MODEL_GLOBALVARS
// #define MODEL_INITIALIZE



void DRIVER_Initialize(
    const int sx, const int sy, const int sz, const int bord,
    HostViewFloat1D vpz, HostViewFloat1D vsv, HostViewFloat1D epsilon,
    HostViewFloat1D delta, HostViewFloat1D phi, HostViewFloat1D theta,

    HostViewFloat1D ch1dxx, HostViewFloat1D ch1dyy, HostViewFloat1D ch1dzz,
    HostViewFloat1D ch1dxy, HostViewFloat1D ch1dyz, HostViewFloat1D ch1dxz,
    HostViewFloat1D v2px, HostViewFloat1D v2pz, HostViewFloat1D v2sz, HostViewFloat1D v2pn,

    DeviceViewFloat1D& dev_ch1dxx, DeviceViewFloat1D& dev_ch1dyy, DeviceViewFloat1D& dev_ch1dzz,
    DeviceViewFloat1D& dev_ch1dxy, DeviceViewFloat1D& dev_ch1dyz, DeviceViewFloat1D& dev_ch1dxz,
    DeviceViewFloat1D& dev_v2px, DeviceViewFloat1D& dev_v2pz, DeviceViewFloat1D& dev_v2sz, DeviceViewFloat1D& dev_v2pn,

    DeviceViewFloat1D& dev_pp, DeviceViewFloat1D& dev_pc,
    DeviceViewFloat1D& dev_qp, DeviceViewFloat1D& dev_qc)
{
    int msize_vol = sx*sy*sz;
    ch1dxx = Kokkos::View<float*, HostMemSpace>("ch1dxx", msize_vol);
    ch1dyy = Kokkos::View<float*, HostMemSpace>("ch1dyy", msize_vol);
    ch1dzz = Kokkos::View<float*, HostMemSpace>("ch1dzz", msize_vol);
    ch1dxy = Kokkos::View<float*, HostMemSpace>("ch1dxy", msize_vol);
    ch1dyz = Kokkos::View<float*, HostMemSpace>("ch1dyz", msize_vol);
    ch1dxz = Kokkos::View<float*, HostMemSpace>("ch1dxz", msize_vol);
    for (int i=0; i<msize_vol; i++) {
        float sinTheta=sin(theta(i));
        float cosTheta=cos(theta(i));
        float sin2Theta=sin(2.0*theta(i));
        float sinPhi=sin(phi(i));
        float cosPhi=cos(phi(i));
        float sin2Phi=sin(2.0*phi(i));
        ch1dxx(i)=sinTheta*sinTheta * cosPhi*cosPhi;
        ch1dyy(i)=sinTheta*sinTheta * sinPhi*sinPhi;
        ch1dzz(i)=cosTheta*cosTheta;
        ch1dxy(i)=sinTheta*sinTheta * sin2Phi;
        ch1dyz(i)=sin2Theta         * sinPhi;
        ch1dxz(i)=sin2Theta         * cosPhi;
    }
#ifdef _DUMP
{
const int iPrint = ind(bord+1,bord+1,bord+1);
printf("ch1dxx=%f; ch1dyy=%f; ch1dzz=%f; ch1dxy=%f; ch1dxz=%f; ch1dyz=%f\n",
ch1dxx(iPrint), ch1dyy(iPrint), ch1dzz(iPrint), ch1dxy(iPrint), ch1dxz(iPrint), ch1dyz(iPrint));
}
#endif

    // coeficients of H1 and H2 at PDEs

    v2px = Kokkos::View<float*, HostMemSpace>("v2px", msize_vol);
    v2pz = Kokkos::View<float*, HostMemSpace>("v2pz", msize_vol);
    v2sz = Kokkos::View<float*, HostMemSpace>("v2sz", msize_vol);
    v2pn = Kokkos::View<float*, HostMemSpace>("v2pn", msize_vol);
    for (int i=0; i<msize_vol; i++){
        v2sz(i)=vsv(i)*vsv(i);
        v2pz(i)=vpz(i)*vpz(i);
        v2px(i)=v2pz(i)*(1.0+2.0*epsilon(i));
        v2pn(i)=v2pz(i)*(1.0+2.0*delta(i));
    }

#ifdef _DUMP
{
const int iPrint=ind(bord+1,bord+1,bord+1);
printf("vsv=%e; vpz=%e, v2pz=%e\n",
     vsv(iPrint), vpz(iPrint), v2pz(iPrint));
printf("v2sz=%e; v2pz=%e, v2px=%e, v2pn=%e\n",
     v2sz(iPrint), v2pz(iPrint), v2px(iPrint), v2pn(iPrint));
}
#endif
    printf("Kokkos init\n");
	KOKKOS_Initialize(sx, sy, sz, bord,
	       ch1dxx, ch1dyy, ch1dzz,
	       ch1dxy, ch1dyz, ch1dxz,
	       v2px, v2pz, v2sz, v2pn,
		   dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
		   dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
		   dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
		   dev_pp, dev_pc,
		   dev_qp, dev_qc);
}



void DRIVER_Finalize()
{
	KOKKOS_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, HostViewFloat1D pc, DeviceViewFloat1D dev_pc)
{
	KOKKOS_Update_pointers(sx, sy, sz, pc, dev_pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
                      int offset,
                    DeviceViewFloat1D dev_ch1dxx, DeviceViewFloat1D dev_ch1dyy, DeviceViewFloat1D dev_ch1dzz,
                    DeviceViewFloat1D dev_ch1dxy, DeviceViewFloat1D dev_ch1dyz, DeviceViewFloat1D dev_ch1dxz,
                    DeviceViewFloat1D dev_v2px, DeviceViewFloat1D dev_v2pz, DeviceViewFloat1D dev_v2sz, DeviceViewFloat1D dev_v2pn,
                    DeviceViewFloat1D dev_pp, DeviceViewFloat1D dev_pc,
                    DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc)
{

	// KOKKOS_Propagate also does TimeForward
	   KOKKOS_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it,
						offset,
						dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
						dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
						dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
						dev_pp, dev_pc,
						dev_qp, dev_qc);
}


void DRIVER_InsertSource(float dt, int it, int iSource, int offset, DeviceViewFloat1D p, DeviceViewFloat1D q, float src)
{
	KOKKOS_InsertSource(src, iSource, offset, p, q);
}
