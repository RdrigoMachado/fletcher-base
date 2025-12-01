#include "sycl_insertsource.h"

void SYCL_InsertSource(const float val, const int iSource, int offset, float* dev_pc, float* dev_qc, sycl::queue q)
{
    sycl::range<1> local_range(BSIZE_X);
    sycl::range<1> global_range(BSIZE_X);
    sycl::nd_range<1> execution_range(global_range, local_range);


    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(execution_range, [=](sycl::nd_item<1> item) {

            const int ix = item.get_global_id(0);

            if (ix==0)
            {
                dev_pc[offset + iSource] += val;
                dev_qc[offset + iSource] += val;
            }
        });
    });
    q.wait();

}
