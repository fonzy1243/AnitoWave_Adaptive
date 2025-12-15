#ifndef ANITOWAVE_ADAPTIVE_SMOOTHER_CUH
#define ANITOWAVE_ADAPTIVE_SMOOTHER_CUH

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <MSBG.cuh>

#define OMEGA 0.85714f
#define MAX_BLOCK_ITER 10
#define TARGET_RESIDUAL_REDUCTION 0.33f

struct MortonCompare
{
    const BlockInfo* info;
    MortonCompare(const BlockInfo* _info) : info(_info) {}
    __device__ bool operator()(const uint32_t& a, const uint32_t& b) const
    {
        return info[a].morton < info[b].morton;
    }
};

// __device__ inline float getPressure(
//     MSBG& grid,
//     uint32_t blockIdx,
//     int3 localCoord,
//     int res,
//     int dx, int dy, int dz
//     )
// {
//     int nx = localCoord.x + dx;
//     int ny = localCoord.y + dy;
//     int nz = localCoord.z + dz;
//
//     if (nx >= 0 && nx < res && ny >= 0 && ny < res && nz >= 0 && nz < res)
//     {
//         uint32_t offset = grid.d_cellOffsets[blockIdx];
//         return grid.d_cellData[PRESSURE][offset + cellIndex3D(nx, ny, nz, res)];
//     }
//
//     int3 blockCoord = grid.d_blockInfo[blockIdx].coord;
//
//     // TODO: Handle T-junction interpolation
//     int nBlockIdx = getNeighborBlockIdx(blockCoord, grid.indexDims,
//                                         (nx < 0) ? -1 : (nx >= res ? 1 : 0),
//                                         (ny < 0) ? -1 : (ny >= res ? 1 : 0),
//                                         (nz < 0) ? -1 : (nz >= res ? 1 : 0));
//
//     if (nBlockIdx == -1) return 0.f; // Domain boundary (Dirichlet 0)
//
//     int nnx = (nx + res) % res;
//     int nny = (ny + res) % res;
//     int nnz = (nz + res) % res;
//
//     uint32_t offset = grid.d_cellOffsets[nBlockIdx];
//     return grid.d_cellData[PRESSURE][offset + cellIndex3D(nnx, nny, nnz, res)];
// }

__device__ inline float getPressure(
    MSBG& grid,
    uint32_t blockIdx,
    int3 localCoord,
    int currentRes,
    int dx, int dy, int dz
)
{
    // 1. Compute global block coordinates
    int3 blockCoord = grid.d_blockInfo[blockIdx].coord;

    // 2. Determine target local coordinates (potentially outside current block)
    int nx = localCoord.x + dx;
    int ny = localCoord.y + dy;
    int nz = localCoord.z + dz;

    // 3. Fast Path: If inside current block, return immediately
    if (nx >= 0 && nx < currentRes &&
        ny >= 0 && ny < currentRes &&
        nz >= 0 && nz < currentRes)
    {
        uint32_t offset = grid.d_cellOffsets[blockIdx];
        return grid.d_cellData[PRESSURE][offset + cellIndex3D(nx, ny, nz, currentRes)];
    }

    // 4. Slow Path: Determine which neighbor block we are targeting
    int3 neighborOffset = make_int3(0, 0, 0);

    if (nx < 0) { neighborOffset.x = -1; nx += currentRes; }
    else if (nx >= currentRes) { neighborOffset.x = 1; nx -= currentRes; }

    if (ny < 0) { neighborOffset.y = -1; ny += currentRes; }
    else if (ny >= currentRes) { neighborOffset.y = 1; ny -= currentRes; }

    if (nz < 0) { neighborOffset.z = -1; nz += currentRes; }
    else if (nz >= currentRes) { neighborOffset.z = 1; nz -= currentRes; }

    // 5. Get Neighbor Index
    uint32_t nBlockIdx = getNeighborBlockIdx(blockCoord, grid.indexDims,
                                            neighborOffset.x, neighborOffset.y, neighborOffset.z);

    // Boundary condition: Dirichlet 0 (Air/Open boundary)
    if (nBlockIdx == -1 || nBlockIdx >= grid.numBlocks) return 0.f;

    // 6. Handle Multi-Resolution T-Junctions [cite: 588, 590]
    uint32_t currentLevel = grid.d_refinementMap[blockIdx];
    uint32_t neighborLevel = grid.d_refinementMap[nBlockIdx];

    int neighborRes = c_blockLayouts[neighborLevel].res;
    uint32_t nOffset = grid.d_cellOffsets[nBlockIdx];

    // Case: Resolutions match (Most common)
    if (currentLevel == neighborLevel) {
        return grid.d_cellData[PRESSURE][nOffset + cellIndex3D(nx, ny, nz, neighborRes)];
    }

    // Case: Neighbor is COARSER (We are fine, they are big)
    // We map our fine coordinate 'nx' to their coarse coordinate.
    // Ratio is always a power of 2 (e.g., 2, 4, 8)
    if (neighborLevel < currentLevel) {
        int scale = 1 << (currentLevel - neighborLevel);

        // When moving to a coarser block, we simply scale down our local coord.
        // Note: We must adjust for the face crossing.
        // If we wrapped around (nx was adjusted by +/- currentRes), we need
        // to ensure we land on the correct side of the coarse block.

        // Re-calculate world-relative offset to handle the wrapping correctly
        // Or simpler: map the "ghost" coordinate relative to the neighbor's origin
        int ghostX = (dx == -1) ? (neighborRes - 1) : (dx == 1) ? 0 : (nx / scale);
        int ghostY = (dy == -1) ? (neighborRes - 1) : (dy == 1) ? 0 : (ny / scale);
        int ghostZ = (dz == -1) ? (neighborRes - 1) : (dz == 1) ? 0 : (nz / scale);

        return grid.d_cellData[PRESSURE][nOffset + cellIndex3D(ghostX, ghostY, ghostZ, neighborRes)];
    }

    // Case: Neighbor is FINER (We are coarse, they are small)
    // We need to grab the specific fine cell that touches our face.
    // This effectively "injects" the fine boundary value into our coarse stencil.
    if (neighborLevel > currentLevel) {
        int scale = 1 << (neighborLevel - currentLevel);

        // Map our coarse coord to the finer grid range
        // If we are accessing the right neighbor (dx=1), we want their x=0 face.
        // If we are accessing the left neighbor (dx=-1), we want their x=max face.
        // If we are not moving in x (dx=0), we scale our current y/z up.

        int ghostX, ghostY, ghostZ;

        // X-dimension logic
        if (dx == -1) ghostX = neighborRes - 1;       // Touching Rightmost face of left neighbor
        else if (dx == 1) ghostX = 0;                 // Touching Leftmost face of right neighbor
        else ghostX = nx * scale + (scale / 2);       // Center alignment (approx)

        // Y-dimension logic
        if (dy == -1) ghostY = neighborRes - 1;
        else if (dy == 1) ghostY = 0;
        else ghostY = ny * scale + (scale / 2);

        // Z-dimension logic
        if (dz == -1) ghostZ = neighborRes - 1;
        else if (dz == 1) ghostZ = 0;
        else ghostZ = nz * scale + (scale / 2);

        return grid.d_cellData[PRESSURE][nOffset + cellIndex3D(ghostX, ghostY, ghostZ, neighborRes)];
    }

    return 0.f; // Should not reach here
}

__global__ inline void initActiveBlocks(
    MSBG grid,
    uint32_t* activeList,
    uint32_t* activeCount,
    uint32_t* blockStatus,
    const uint32_t targetLevel
    )
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid.numBlocks) return;

    // TODO: Fix simplification
    if (const uint32_t level = grid.d_refinementMap[idx]; level <= targetLevel)
    {
        const uint32_t pos = atomicAdd(activeCount, 1);
        activeList[pos] = idx;
        blockStatus[idx] = 1;
    } else
    {
        blockStatus[idx] = 0;
    }
}

__device__ inline void activateBlock(
    uint32_t blockIdx,
    uint32_t* nextList,
    uint32_t* nextCount,
    uint32_t* nextStatus
    )
{
    // Atomic Compare Exchange to ensure we only add it once per pass
    // Simulating CAS on uint8 via casting or logic (using int atomic for safety usually)
    // Here we use a simpler flag check since status is reset every pass
    if (atomicExch(&nextStatus[blockIdx], 1) == 0) {
        uint32_t pos = atomicAdd(nextCount, 1);
        nextList[pos] = blockIdx;
    }
}

__global__ inline void smoothActiveBlocks(
    MSBG grid,
    const uint32_t* activeList,
    uint32_t activeCount,
    uint32_t* nextList,
    uint32_t* nextCount,
    uint32_t* nextStatus,
    int rbPhase,
    float theta
    )
{
    // One CUDA block per MSBG block
    uint32_t listIdx = blockIdx.x;
    if (listIdx >= activeCount) return;

    uint32_t gridBlockIdx = activeList[listIdx];

    BlockInfo info = grid.d_blockInfo[gridBlockIdx];
    if ((info.coord.x + info.coord.y + info.coord.z) % 2 != rbPhase) return;

    uint32_t level = grid.d_refinementMap[gridBlockIdx];
    int res = c_blockLayouts[level].res;
    uint32_t offset = grid.d_cellOffsets[gridBlockIdx];

    float initialBlockRes = -1.f;
    __shared__ float s_maxRes;

    float initialRes = 0.f;

    for (int iter = 0; iter < MAX_BLOCK_ITER; iter++)
    {
        if (threadIdx.x == 0) s_maxRes = 0.f;
        __syncthreads();

        for (int color = 0; color < 2; color++)
        {
            int numCells = res * res * res;
            for (int i = threadIdx.x; i < numCells; i += blockDim.x)
            {
                int z = i / (res * res);
                int y = (i % (res * res)) / res;
                int x = i % res;

                if ((x + y + z) % 2 != color) continue;

                // Coefficients (Beta = 1/rho)
                uint32_t fx = grid.d_faceXOffsets[gridBlockIdx] + faceXIndex3D(x, y, z, res);
                uint32_t fx_p = grid.d_faceXOffsets[gridBlockIdx] + faceXIndex3D(x+1, y, z, res);
                uint32_t fy = grid.d_faceYOffsets[gridBlockIdx] + faceYIndex3D(x, y, z, res);
                uint32_t fy_p = grid.d_faceYOffsets[gridBlockIdx] + faceYIndex3D(x, y+1, z, res);
                uint32_t fz = grid.d_faceZOffsets[gridBlockIdx] + faceZIndex3D(x, y, z, res);
                uint32_t fz_p = grid.d_faceZOffsets[gridBlockIdx] + faceZIndex3D(x, y, z+1, res);

                float bW = grid.d_faceXData[BETA_COEFF_X][fx];
                float bE = grid.d_faceXData[BETA_COEFF_X][fx_p];
                float bS = grid.d_faceYData[BETA_COEFF_Y][fy];
                float bN = grid.d_faceYData[BETA_COEFF_Y][fy_p];
                float bB = grid.d_faceZData[BETA_COEFF_Z][fz];
                float bT = grid.d_faceZData[BETA_COEFF_Z][fz_p];

                float diag = -(bW + bE + bS + bN + bB + bT);
                if (fabsf(diag) < 1e-6f) continue; // Skip empty/air cells if 0

                // Neighbors Pressure
                float pW = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, -1, 0, 0);
                float pE = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, 1, 0, 0);
                float pS = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, 0, -1, 0);
                float pN = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, 0, 1, 0);
                float pB = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, 0, 0, -1);
                float pT = getPressure(grid, gridBlockIdx, make_int3(x, y, z), res, 0, 0, 1);

                float Ax = bW*pW + bE*pE + bS*pS + bN*pN + bB*pB + bT*pT + diag * grid.d_cellData[PRESSURE][offset + i];
                float rhs = grid.d_cellData[DIVERGENCE][offset + i];

                // Residual r = b - Ax
                float r = rhs - Ax;

                // Update: delta = omega * r / diag
                // Note: The paper uses algebraic aggregation where diag elements are sum of coeffs
                float delta = OMEGA * (r / diag);

                grid.d_cellData[PRESSURE][offset + i] += delta;

                // Track max residual
                float absR = fabsf(r);
                auto address = (unsigned int*)&s_maxRes;
                unsigned int val = __float_as_uint(absR);
                atomicMax(address, val);
            }
            __syncthreads();

            if (threadIdx.x == 0)
            {
                if (iter == 0) initialBlockRes = s_maxRes;
            }
            __syncthreads();

            if (initialBlockRes > 1e-9f && s_maxRes < (initialBlockRes * TARGET_RESIDUAL_REDUCTION)) break;
        }
    }

    if (threadIdx.x == 0 && s_maxRes > theta)
    {
        activateBlock(gridBlockIdx, nextList, nextCount, nextStatus);

        int3 c = info.coord;
        int3 nCoords[6] = {
            make_int3(c.x-1, c.y, c.z), make_int3(c.x+1, c.y, c.z),
            make_int3(c.x, c.y-1, c.z), make_int3(c.x, c.y+1, c.z),
            make_int3(c.x, c.y, c.z-1), make_int3(c.x, c.y, c.z+1)
        };

        for (int k = 0; k < 6; k++)
        {
            uint32_t nIdx = getNeighborBlockIdx(nCoords[k], grid.indexDims, 0, 0, 0);
            if (nIdx != -1)
            {
                activateBlock(nIdx, nextList, nextCount, nextStatus);
            }
        }
    }
}

class AdaptiveSmoother
{
public:
    uint32_t* d_activeList[2];
    uint32_t* d_activeCount[2];
    uint32_t* d_blockStatus[2];

    uint32_t numBlocks;

    AdaptiveSmoother(uint32_t n) : numBlocks(n)
    {
        cudaMalloc(&d_activeList[0], numBlocks * sizeof(uint32_t));
        cudaMalloc(&d_activeList[1], numBlocks * sizeof(uint32_t));

        cudaMalloc(&d_activeCount[0], sizeof(uint32_t));
        cudaMalloc(&d_activeCount[1], sizeof(uint32_t));

        cudaMalloc(&d_blockStatus[0], numBlocks * sizeof(uint32_t));
        cudaMalloc(&d_blockStatus[1], numBlocks * sizeof(uint32_t));
    }

    ~AdaptiveSmoother()
    {
        cudaFree(d_activeList[0]);
        cudaFree(d_activeList[1]);

        cudaFree(d_activeCount[0]);
        cudaFree(d_activeCount[1]);

        cudaFree(d_blockStatus[0]);
        cudaFree(d_blockStatus[1]);
    }

    void solve(MSBGManager& mg, int maxGlobalIter, float theta)
    {
        int curr = 0;
        int next = 1;

        cudaMemset(d_activeCount[curr], 0, sizeof(uint32_t));
        cudaMemset(d_blockStatus[curr], 0, numBlocks * sizeof(uint32_t));

        int threads = 256;
        int blocks = (numBlocks + threads - 1) / threads;

        initActiveBlocks<<<blocks, threads>>>(
            mg.grid, d_activeList[curr], d_activeCount[curr], d_blockStatus[curr], MAX_LEVELS-1
        );
        cudaDeviceSynchronize();

        for (int iter = 0; iter < maxGlobalIter; iter++)
        {
            uint32_t h_count = 0;
            cudaMemcpy(&h_count, d_activeCount[curr], sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (h_count == 0) break;

            // Sort by Morton code for memory coherence
            thrust::device_ptr<uint32_t> ptr(d_activeList[curr]);
            MortonCompare comp(mg.grid.d_blockInfo);
            thrust::sort(thrust::device, ptr, ptr + h_count, comp);

            cudaMemset(d_activeCount[next], 0, sizeof(uint32_t));
            cudaMemset(d_blockStatus[next], 0, numBlocks * sizeof(uint32_t));

            for (int rb = 0; rb < 2; rb++)
            {
                smoothActiveBlocks<<<h_count, 256>>>(
                    mg.grid,
                    d_activeList[curr],
                    h_count,
                    d_activeList[next],
                    d_activeCount[next],
                    d_blockStatus[next],
                    rb,
                    theta
                );
            }

            std::swap(curr, next);
        }
    }
};

#endif //ANITOWAVE_ADAPTIVE_SMOOTHER_CUH