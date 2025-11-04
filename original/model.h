#ifndef _MODEL
#define _MODEL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"

#ifdef PAPI
#include "ModPAPI.h"
#endif

#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#include "KOKKOS/kokkos_defines.h"

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr,
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it,
	   HostViewFloat1D pp, HostViewFloat1D pc, HostViewFloat1D qp, HostViewFloat1D qc,
	   HostViewFloat1D vpz, HostViewFloat1D vsv, HostViewFloat1D epsilon, HostViewFloat1D delta,
	   HostViewFloat1D phi, HostViewFloat1D theta, int absorb);

#endif
