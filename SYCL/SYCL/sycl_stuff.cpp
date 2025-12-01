#include "sycl_stuff.h"

static size_t sxsy=0;

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
    sycl::queue q)
{

    // Check sx,sy values
    if (sx%BSIZE_X != 0)
    {
        printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
        exit(1);
    }
    if (sy%BSIZE_Y != 0)
    {
        printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
        exit(1);
    }

    sxsy=sx*sy; // one plan
    const size_t msize_vol=sxsy*sz;
    const size_t msize_vol_bytes=msize_vol*sizeof(float);
    const size_t msize_vol_extra=msize_vol+2*sxsy; // 2 extra plans for wave fields

    dev_ch1dxx = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dxx, ch1dxx, msize_vol_bytes).wait();
    dev_ch1dyy = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dyy, ch1dyy, msize_vol_bytes).wait();
    dev_ch1dzz = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dzz, ch1dzz, msize_vol_bytes).wait();
    dev_ch1dxy = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dxy, ch1dxy, msize_vol_bytes).wait();
    dev_ch1dyz = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dyz, ch1dyz, msize_vol_bytes).wait();
    dev_ch1dxz = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_ch1dxz, ch1dxz, msize_vol_bytes).wait();
    dev_v2px = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_v2px, v2px, msize_vol_bytes).wait();
    dev_v2pz = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_v2pz, v2pz, msize_vol_bytes).wait();
    dev_v2sz = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_v2sz, v2sz, msize_vol_bytes).wait();
    dev_v2pn = sycl::malloc_device<float>(msize_vol, q);
    q.memcpy(dev_v2pn, v2pn, msize_vol_bytes).wait();

    // Wave field arrays with an extra plan
    dev_pp = sycl::malloc_device<float>(msize_vol_extra, q);
    dev_pc = sycl::malloc_device<float>(msize_vol_extra, q);
    dev_qp = sycl::malloc_device<float>(msize_vol_extra, q);
    dev_qc = sycl::malloc_device<float>(msize_vol_extra, q);

    q.fill(dev_pp, 0, msize_vol_extra).wait();
    q.fill(dev_pc, 0, msize_vol_extra).wait();
    q.fill(dev_qp, 0, msize_vol_extra).wait();
    q.fill(dev_qc, 0, msize_vol_extra).wait();

    dev_pp+=sxsy;
    dev_pc+=sxsy;
    dev_qp+=sxsy;
    dev_qc+=sxsy;

    printf("GPU memory usage = %ld MiB\n", 15*msize_vol_bytes/1024/1024);
}


void SYCL_Finalize()
{
    printf("SYCL_Finalize: SUCCESS\n");
}



void SYCL_Update_pointers(const int sx, const int sy, const int sz, float* pc, float* dev_pc, sycl::queue q)
{
    size_t total_elements = (size_t)sx * (size_t)sy * (size_t)sz;
    size_t msize_vol_bytes = total_elements * sizeof(float);
    // q.memcpy(pc, dev_pc, msize_vol_bytes).wait();
}
