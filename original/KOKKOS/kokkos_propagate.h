#ifndef __KOKKOS_PROPAGATE
#define __KOKKOS_PROPAGATE

#include "kokkos_defines.h"
#include "kokkos_propagate.h"
#include "../derivatives.h"

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void KOKKOS_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it,
			int offset,
	       HostViewFloat1D pp, HostViewFloat1D pc, HostViewFloat1D qp, HostViewFloat1D qc,
		   DeviceViewFloat1D dev_ch1dxx, DeviceViewFloat1D dev_ch1dyy, DeviceViewFloat1D dev_ch1dzz,
		   DeviceViewFloat1D dev_ch1dxy, DeviceViewFloat1D dev_ch1dyz, DeviceViewFloat1D dev_ch1dxz,
		   DeviceViewFloat1D dev_v2px, DeviceViewFloat1D dev_v2pz, DeviceViewFloat1D dev_v2sz, DeviceViewFloat1D dev_v2pn,
		   DeviceViewFloat1D dev_pp, DeviceViewFloat1D dev_pc,
		   DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc);

void KOKKOS_SwapArrays(HostViewFloat1D pp, HostViewFloat1D pc, HostViewFloat1D qp, HostViewFloat1D qc);

#endif
