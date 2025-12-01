#ifndef _BOUNDARY
#define _BOUNDARY
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "map.h"
#include "KOKKOS/kokkos_defines.h"

#define FRACABS 0.03125


// RandomVelocityBoundary: creates a boundary with random velocity around domain


void RandomVelocityBoundary(int sx, int sy, int sz,
			    int nx, int ny, int nz,
			    int bord, int absorb,
			    Kokkos::View<float *, HostMemSpace> vpz, Kokkos::View<float *, HostMemSpace> vsv);

#endif
