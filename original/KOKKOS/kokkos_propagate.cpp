#include "kokkos_propagate.h"

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
			   DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc)
{

  dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
  dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);

  Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecutionSpace>(numBlocks.x * numBlocks.y, threadsPerBlock.x * threadsPerBlock.y),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team){
            //mapping to cuda logic
            int block_id = team.league_rank();
            int blockIdX = block_id % numBlocks.x;
            int blockIdY = block_id / numBlocks.y;

            int thread_id = team.team_rank();
            int threadIdX = thread_id % threadsPerBlock.x;
            int threadIdY = thread_id / threadsPerBlock.y;

            const int ix=blockIdX * threadsPerBlock.x + threadIdX;
            const int iy=blockIdY * threadsPerBlock.y + threadIdY;

            const int strideX=ind(1,0,0)-ind(0,0,0);
            const int strideY=ind(0,1,0)-ind(0,0,0);
            const int strideZ=ind(0,0,1)-ind(0,0,0);

            const float dxxinv=1.0f/(dx*dx);
            const float dyyinv=1.0f/(dy*dy);
            const float dzzinv=1.0f/(dz*dz);
            const float dxyinv=1.0f/(dx*dy);
            const float dxzinv=1.0f/(dx*dz);
            const float dyzinv=1.0f/(dy*dz);

            // solve both equations in all internal grid points,
            // including absortion zone

            for (int iz=bord+1; iz<sz-bord-1; iz++) {

                const int i=ind(ix,iy,iz);

                // p derivatives, H1(p) and H2(p)

                const float pyy= Der2(pc, i + offset, strideY, dyyinv);
                const float pzz= Der2(pc, i + offset, strideZ, dzzinv);
                const float pxx= Der2(pc, i + offset, strideX, dxxinv);
                const float pxy= DerCross(pc, i + offset, strideX, strideY, dxyinv);
                const float pyz= DerCross(pc, i + offset, strideY, strideZ, dyzinv);
                const float pxz= DerCross(pc, i + offset, strideX, strideZ, dxzinv);

                const float cpxx=dev_ch1dxx(i)*pxx;
                const float cpyy=dev_ch1dyy(i)*pyy;
                const float cpzz=dev_ch1dzz(i)*pzz;
                const float cpxy=dev_ch1dxy(i)*pxy;
                const float cpxz=dev_ch1dxz(i)*pxz;
                const float cpyz=dev_ch1dyz(i)*pyz;
                const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
                const float h2p=pxx+pyy+pzz-h1p;

                // q derivatives, H1(q) and H2(q)

                const float qxx= Der2(qc, i + offset, strideX, dxxinv);
                const float qyy= Der2(qc, i + offset, strideY, dyyinv);
                const float qzz= Der2(qc, i + offset, strideZ, dzzinv);
                const float qxy= DerCross(qc, i + offset, strideX,  strideY, dxyinv);
                const float qyz= DerCross(qc, i + offset, strideY,  strideZ, dyzinv);
                const float qxz= DerCross(qc, i + offset, strideX,  strideZ, dxzinv);

                const float cqxx=dev_ch1dxx(i)*qxx;
                const float cqyy=dev_ch1dyy(i)*qyy;
                const float cqzz=dev_ch1dzz(i)*qzz;
                const float cqxy=dev_ch1dxy(i)*qxy;
                const float cqxz=dev_ch1dxz(i)*qxz;
                const float cqyz=dev_ch1dyz(i)*qyz;
                const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
                const float h2q=qxx+qyy+qzz-h1q;

                // p-q derivatives, H1(p-q) and H2(p-q)

                const float h1pmq=h1p-h1q;
                const float h2pmq=h2p-h2q;

                // rhs of p and q equations

                const float rhsp=dev_v2px(i)*h2p + dev_v2pz(i)*h1q + dev_v2sz(i)*h1pmq;
                const float rhsq=dev_v2pn(i)*h2p + dev_v2pz(i)*h1q - dev_v2sz(i)*h2pmq;

                // new p and q

                pp(i + offset)=2.0f*pc(i + offset) - pp(i + offset) + rhsp*dt*dt;
                qp(i + offset)=2.0f*qc(i + offset) - qp(i + offset) + rhsq*dt*dt;

            }
        }
  );

  KOKKOS_SwapArrays(dev_pp, dev_pc, dev_qp, dev_qc);
  Kokkos::fence();

}

// swap array pointers on time forward array propagation
void KOKKOS_SwapArrays(DeviceViewFloat1D dev_pp, DeviceViewFloat1D dev_pc, DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc) {
  auto temp = dev_pp;
  dev_pp = dev_pc;
  dev_pc = temp;

  temp = dev_qp;
  dev_qp = dev_qc;
  dev_qc = temp;
}
