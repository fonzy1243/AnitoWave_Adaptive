#ifndef ANITOWAVE_ADAPTIVE_MULTIGRIDTRANSFER_CUH
#define ANITOWAVE_ADAPTIVE_MULTIGRIDTRANSFER_CUH

#include <MSBG.cuh>
#include <cuda/std/cstdint>

// Restriction: Fine -> Coarse
// R = P^T implies summation if P is injection
__global__ void restrictResidualsKernel(
    uint32_t numBlocks,
    const uint32_t* __restrict__ refinementMap,
    const uint32_t* __restrict__ fineOffsets,
    const uint32_t* __restrict__ coarseOffsets,
    const float* __restrict__ fineResidual,
    float* __restrict__ coarseResidual,
    int mgLevel
    )
{
    uint32_t bIdx = blockIdx.x;
    if (bIdx >= numBlocks) return;

    uint32_t msbgLevel = refinementMap[bIdx];

    int fineRes = c_blockLayouts[msbgLevel].res >> mgLevel;

    if (fineRes <= 1) return;

    int coarseRes = fineRes / 2;

    uint32_t fineBlockOffset = fineOffsets[bIdx];
    uint32_t coarseBlockOffset = coarseOffsets[bIdx];

    for (int i = threadIdx.x; i < coarseRes * coarseRes * coarseRes; i += blockDim.x)
    {
        int z = i / (coarseRes * coarseRes);
        int rem = i % (coarseRes * coarseRes);
        int y = rem / coarseRes;
        int x = rem % coarseRes;

        // Sum 8 fine children (Algebraic Aggregation / Piecewise Constant)
        float sum = 0.0f;

        // Base fine coordinates
        int fx = x * 2;
        int fy = y * 2;
        int fz = z * 2;

        sum += fineResidual[fineBlockOffset + cellIndex3D(fx,   fy,   fz,   fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx+1, fy,   fz,   fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx,   fy+1, fz,   fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx+1, fy+1, fz,   fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx,   fy,   fz+1, fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx+1, fy,   fz+1, fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx,   fy+1, fz+1, fineRes)];
        sum += fineResidual[fineBlockOffset + cellIndex3D(fx+1, fy+1, fz+1, fineRes)];

        coarseResidual[coarseBlockOffset + i] = sum;
    }
}

// Prolongation: Coarse -> Fine (Injection/Correction)
// e_fine += P * e_coarse
__global__ void prolongateCorrection(
    uint32_t numBlocks,
    const uint32_t* __restrict__ refinementMap,
    const uint32_t* __restrict__ fineOffsets,
    const uint32_t* __restrict__ coarseOffsets,
    float* __restrict__ fineCorrection,
    const float* __restrict__ coarseCorrection,
    int mgLevel
) {
    uint32_t bIdx = blockIdx.x;
    if (bIdx >= numBlocks) return;

    uint32_t msbgLevel = refinementMap[bIdx];
    int fineRes = c_blockLayouts[msbgLevel].res >> mgLevel;

    if (fineRes <= 1) return;

    int coarseRes = fineRes / 2;

    uint32_t fineBlockOffset = fineOffsets[bIdx];
    uint32_t coarseBlockOffset = coarseOffsets[bIdx];

    // One thread per FINE cell (more parallelism)
    for (int i = threadIdx.x; i < fineRes * fineRes * fineRes; i += blockDim.x) {
        int z = i / (fineRes * fineRes);
        int rem = i % (fineRes * fineRes);
        int y = rem / fineRes;
        int x = rem % fineRes;

        // Find parent in coarse grid
        int cx = x / 2;
        int cy = y / 2;
        int cz = z / 2;

        float coarseVal = coarseCorrection[coarseBlockOffset + cellIndex3D(cx, cy, cz, coarseRes)];

        // Add correction (V-cycle step: x = x + P*e)
        fineCorrection[fineBlockOffset + i] += coarseVal;
    }
}

class MultigridTransfer
{
public:
    void restrictResiduals(
        MSBGManager& mg,
        float* d_fineRes,
        float* d_coarseRes,
        uint32_t* d_fineOffsets,
        uint32_t* d_coarseOffsets,
        int mgLevel
        )
    {
        int threads = 256;
        restrictResidualsKernel<<<mg.grid.numBlocks, threads>>>(
            mg.grid.numBlocks,
            mg.grid.d_refinementMap,
            d_fineOffsets,
            d_coarseOffsets,
            d_fineRes,
            d_coarseRes,
            mgLevel
        );
        cudaDeviceSynchronize();
    }

    void prolongateAndCorrect(
        MSBGManager& mg,
        float* d_fineCorr,
        float* d_coarseCorr,
        uint32_t* d_fineOffsets,
        uint32_t* d_coarseOffsets,
        int mgLevel
        )
    {
        int threads = 256;
        prolongateCorrection<<<mg.grid.numBlocks, threads>>>(
            mg.grid.numBlocks,
            mg.grid.d_refinementMap,
            d_fineOffsets,
            d_coarseOffsets,
            d_fineCorr,
            d_coarseCorr,
            mgLevel
        );
        cudaDeviceSynchronize();
    }
};

#endif //ANITOWAVE_ADAPTIVE_MULTIGRIDTRANSFER_CUH