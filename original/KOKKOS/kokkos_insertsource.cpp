#include "kokkos_defines.h"
#include "kokkos_insertsource.h"


void KOKKOS_InsertSource(const float val, const int iSource, int offset, DeviceViewFloat1D dev_pc, DeviceViewFloat1D dev_qc)
{
    dim3 threadsPerBlock(BSIZE_X, 1);
    dim3 numBlocks(1,1);

    Kokkos::parallel_for("KOKKOS_InsertSource",
        Kokkos::TeamPolicy<ExecutionSpace>(numBlocks.x * numBlocks.y, threadsPerBlock.x * threadsPerBlock.y),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {

            int block_id = team.league_rank();
            int blockIdX = block_id % numBlocks.x;

            int thread_id = team.team_rank();
            int threadIdX = thread_id % threadsPerBlock.x;

            const int ix = blockIdX * threadsPerBlock.x + threadIdX;

            if (ix==0)
            {
                dev_pc(offset + iSource) += val;
                dev_qc(offset + iSource) += val;
            }
        }
    );
    Kokkos::fence();

}
