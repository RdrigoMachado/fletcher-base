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
#include <sycl/sycl.hpp>

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr,
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it,
	   float* pp, float* pc, float* qp, float* qc,
	   float* vpz, float* vsv, float* epsilon, float* delta,
	   float* phi, float* theta, int absorb, sycl::queue q);

#endif
