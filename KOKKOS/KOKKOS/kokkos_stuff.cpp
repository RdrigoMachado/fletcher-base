#include "kokkos_stuff.h"

static size_t sxsy=0;

void KOKKOS_Initialize(const int sx, const int sy, const int sz, const int bord,
                        HostViewFloat1D ch1dxx, HostViewFloat1D ch1dyy, HostViewFloat1D ch1dzz, HostViewFloat1D ch1dxy,
                        HostViewFloat1D ch1dyz, HostViewFloat1D ch1dxz,
                        HostViewFloat1D v2px, HostViewFloat1D v2pz, HostViewFloat1D v2sz, HostViewFloat1D v2pn,

                        DeviceViewFloat1D& dev_ch1dxx, DeviceViewFloat1D& dev_ch1dyy, DeviceViewFloat1D& dev_ch1dzz,
                        DeviceViewFloat1D& dev_ch1dxy, DeviceViewFloat1D& dev_ch1dyz, DeviceViewFloat1D& dev_ch1dxz,
                        DeviceViewFloat1D& dev_v2px, DeviceViewFloat1D& dev_v2pz, DeviceViewFloat1D& dev_v2sz, DeviceViewFloat1D& dev_v2pn,
                        DeviceViewFloat1D& dev_pp, DeviceViewFloat1D& dev_pc,
                        DeviceViewFloat1D& dev_qp, DeviceViewFloat1D& dev_qc)
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

   dev_ch1dxx = Kokkos::View<float*>("dev_ch1dxx", msize_vol);
   Kokkos::deep_copy(dev_ch1dxx, ch1dxx);
   dev_ch1dyy = Kokkos::View<float*>("dev_ch1dyy", msize_vol);
   Kokkos::deep_copy(dev_ch1dyy, ch1dyy);
   dev_ch1dzz = Kokkos::View<float*>("dev_ch1dzz", msize_vol);
   Kokkos::deep_copy(dev_ch1dzz, ch1dzz);
   dev_ch1dxy = Kokkos::View<float*>("dev_ch1dxy", msize_vol);
   Kokkos::deep_copy(dev_ch1dxy, ch1dxy);
   dev_ch1dyz = Kokkos::View<float*>("dev_ch1dyz", msize_vol);
   Kokkos::deep_copy(dev_ch1dyz, ch1dyz);
   dev_ch1dxz = Kokkos::View<float*>("dev_ch1dxz", msize_vol);
   Kokkos::deep_copy(dev_ch1dxz, ch1dxz);
   dev_v2px = Kokkos::View<float*>("dev_v2px", msize_vol);
   Kokkos::deep_copy(dev_v2px, v2px);
   dev_v2pz = Kokkos::View<float*>("dev_v2pz", msize_vol);
   Kokkos::deep_copy(dev_v2pz, v2pz);
   dev_v2sz = Kokkos::View<float*>("dev_v2sz", msize_vol);
   Kokkos::deep_copy(dev_v2sz, v2sz);
   dev_v2pn = Kokkos::View<float*>("dev_v2pn", msize_vol);
   Kokkos::deep_copy(dev_v2pn, v2pn);

   // Wave field arrays with an extra plan
   dev_pp = Kokkos::View<float*, DeviceMemSpace>("dev_pp", msize_vol_extra);
   dev_pc = Kokkos::View<float*, DeviceMemSpace>("dev_pc", msize_vol_extra);
   dev_qp = Kokkos::View<float*, DeviceMemSpace>("dev_qp", msize_vol_extra);
   dev_qc = Kokkos::View<float*, DeviceMemSpace>("dev_qc", msize_vol_extra);

   printf("Parallel\n");

   Kokkos::parallel_for("SetElementsToZero",
        msize_vol_extra,
        KOKKOS_LAMBDA (const int i) {
            dev_pp(i) = 0;
            dev_pc(i) = 0;
            dev_qp(i) = 0;
            dev_qc(i) = 0;
        }
   );

   // dev_pp+=sxsy;
   // dev_pc+=sxsy;
   // dev_qp+=sxsy;
   // dev_qc+=sxsy;

  printf("GPU memory usage = %ld MiB\n", 15*msize_vol_bytes/1024/1024);

}


void KOKKOS_Finalize()
{
   printf("KOKKOS_Finalize: SUCCESS\n");
}



void KOKKOS_Update_pointers(const int sx, const int sy, const int sz, HostViewFloat1D pc, DeviceViewFloat1D dev_pc)
{
    size_t sxsy = sx * sy;
    size_t msize_vol = sx * sy * sz;

    auto dev_sub = Kokkos::subview(dev_pc, std::make_pair(sxsy, sxsy + msize_vol));

    Kokkos::deep_copy(pc, dev_sub);
}
