#include "sycl_propagate.h"

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void SYCL_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it,
			int offset,
		    float* pp, float* pc, float* qp, float* qc,
			   float* dev_ch1dxx, float* dev_ch1dyy, float* dev_ch1dzz,
			   float* dev_ch1dxy, float* dev_ch1dyz, float* dev_ch1dxz,
			   float* dev_v2px, float* dev_v2pz, float* dev_v2sz, float* dev_v2pn,
			   float* dev_pp, float* dev_pc,
			   float* dev_qp, float* dev_qc, sycl::queue q)
{

    dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
    dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);

    sycl::range<2> local_range(BSIZE_Y, BSIZE_X);
    sycl::range<2> global_range(sy, sx);
    sycl::nd_range<2> execution_range(global_range, local_range);

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(execution_range, [=](sycl::nd_item<2> item) {

            const int ix = item.get_global_id(1);
            const int iy = item.get_global_id(0);

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

                const float cpxx=dev_ch1dxx[i]*pxx;
                const float cpyy=dev_ch1dyy[i]*pyy;
                const float cpzz=dev_ch1dzz[i]*pzz;
                const float cpxy=dev_ch1dxy[i]*pxy;
                const float cpxz=dev_ch1dxz[i]*pxz;
                const float cpyz=dev_ch1dyz[i]*pyz;
                const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
                const float h2p=pxx+pyy+pzz-h1p;

                // q derivatives, H1(q) and H2(q)

                const float qxx= Der2(qc, i + offset, strideX, dxxinv);
                const float qyy= Der2(qc, i + offset, strideY, dyyinv);
                const float qzz= Der2(qc, i + offset, strideZ, dzzinv);
                const float qxy= DerCross(qc, i + offset, strideX,  strideY, dxyinv);
                const float qyz= DerCross(qc, i + offset, strideY,  strideZ, dyzinv);
                const float qxz= DerCross(qc, i + offset, strideX,  strideZ, dxzinv);

                const float cqxx=dev_ch1dxx[i]*qxx;
                const float cqyy=dev_ch1dyy[i]*qyy;
                const float cqzz=dev_ch1dzz[i]*qzz;
                const float cqxy=dev_ch1dxy[i]*qxy;
                const float cqxz=dev_ch1dxz[i]*qxz;
                const float cqyz=dev_ch1dyz[i]*qyz;
                const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
                const float h2q=qxx+qyy+qzz-h1q;

                // p-q derivatives, H1(p-q) and H2(p-q)

                const float h1pmq=h1p-h1q;
                const float h2pmq=h2p-h2q;

                // rhs of p and q equations

                const float rhsp=dev_v2px[i]*h2p + dev_v2pz[i]*h1q + dev_v2sz[i]*h1pmq;
                const float rhsq=dev_v2pn[i]*h2p + dev_v2pz[i]*h1q - dev_v2sz[i]*h2pmq;

                // new p and q

                pp[i + offset]=2.0f*pc[i + offset] - pp[i + offset] + rhsp*dt*dt;
                qp[i + offset]=2.0f*qc[i + offset] - qp[i + offset] + rhsq*dt*dt;

            }
        });
    });

  SYCL_SwapArrays(dev_pp, dev_pc, dev_qp, dev_qc);
  q.wait();
}

// swap array pointers on time forward array propagation
void SYCL_SwapArrays(float*& dev_pp, float*& dev_pc,
                     float*& dev_qp, float*& dev_qc)
{
    std::swap(dev_pp, dev_pc);
    std::swap(dev_qp, dev_qc);
}
