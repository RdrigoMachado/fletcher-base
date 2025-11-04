#ifndef _KOKKOS_STUFF
#define _KOKKOS_STUFF

#include "kokkos_defines.h"

void KOKKOS_Initialize(const int sx, const int sy, const int sz, const int bord,
	       HostViewFloat1D ch1dxx, HostViewFloat1D ch1dyy, HostViewFloat1D ch1dzz,
	       HostViewFloat1D ch1dxy, HostViewFloat1D ch1dyz, HostViewFloat1D ch1dxz,
	       HostViewFloat1D v2px, HostViewFloat1D v2pz, HostViewFloat1D v2sz, HostViewFloat1D v2pn,
		   DeviceViewFloat1D dev_ch1dxx, DeviceViewFloat1D dev_ch1dyy, DeviceViewFloat1D dev_ch1dzz,
		   DeviceViewFloat1D dev_ch1dxy, DeviceViewFloat1D dev_ch1dyz, DeviceViewFloat1D dev_ch1dxz,
		   DeviceViewFloat1D dev_v2px, DeviceViewFloat1D dev_v2pz, DeviceViewFloat1D dev_v2sz, DeviceViewFloat1D dev_v2pn,
		   DeviceViewFloat1D dev_pp, DeviceViewFloat1D dev_pc,
		   DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc);

void KOKKOS_Finalize();

void KOKKOS_Update_pointers(const int sx, const int sy, const int sz, HostViewFloat1D pc, DeviceViewFloat1D dev_pc);

#endif
