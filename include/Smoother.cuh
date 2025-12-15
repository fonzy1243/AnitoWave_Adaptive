#ifndef ANITOWAVE_ADAPTIVE_SMOOTHER_CUH
#define ANITOWAVE_ADAPTIVE_SMOOTHER_CUH

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <MSBG.cuh>

#define OMEGA 0.85714f
#define MAX_BLOCK_ITER 10
#define TARGET_RESIDUAL_REDUCTION 0.33f

struct MortonCompare {
    const BlockInfo* info;
    MortonCompare(const BlockInfo* _info) : info(_info) {}
    __device__ bool operator()(const uint32_t& a, const uint32_t& b) const {
        return info[a].morton < info[b].morton;
    }
};

// --- ACCESS HELPER (Explicit Pointers) ---

__device__ inline float getPressureLevel(
    MSBG& grid,
    const float* __restrict__ p_data,      // Explicit Pressure
    const uint32_t* __restrict__ offsets,  // Explicit Offsets
    uint32_t blockIdx,
    int3 localCoord,
    int currentRes,
    int dx, int dy, int dz
) {
    int3 blockCoord = grid.d_blockInfo[blockIdx].coord;
    int nx = localCoord.x + dx;
    int ny = localCoord.y + dy;
    int nz = localCoord.z + dz;

    // 1. Fast Path
    if (nx >= 0 && nx < currentRes && ny >= 0 && ny < currentRes && nz >= 0 && nz < currentRes) {
        uint32_t offset = offsets[blockIdx];
        return p_data[offset + cellIndex3D(nx, ny, nz, currentRes)];
    }

    // 2. Neighbor Lookup
    int3 neighborOffset = make_int3(0, 0, 0);
    if (nx < 0) { neighborOffset.x = -1; nx += currentRes; }
    else if (nx >= currentRes) { neighborOffset.x = 1; nx -= currentRes; }

    if (ny < 0) { neighborOffset.y = -1; ny += currentRes; }
    else if (ny >= currentRes) { neighborOffset.y = 1; ny -= currentRes; }

    if (nz < 0) { neighborOffset.z = -1; nz += currentRes; }
    else if (nz >= currentRes) { neighborOffset.z = 1; nz -= currentRes; }

    uint32_t nBlockIdx = getNeighborBlockIdx(blockCoord, grid.indexDims, neighborOffset.x, neighborOffset.y, neighborOffset.z);

    // Dirichlet BC (Air)
    if (nBlockIdx == -1 || nBlockIdx >= grid.numBlocks) return 0.f;

    // 3. Ghost Value Retrieval
    uint32_t currentLevel = grid.d_refinementMap[blockIdx];
    uint32_t neighborLevel = grid.d_refinementMap[nBlockIdx];
    uint32_t nOffset = offsets[nBlockIdx];

    // Determine neighbor's resolution AT THIS MG LEVEL
    // If levels match, resolution matches.
    if (currentLevel == neighborLevel) {
        return p_data[nOffset + cellIndex3D(nx, ny, nz, currentRes)];
    }

    // T-Junction Logic (Coarse-Fine / Fine-Coarse)
    // If neighbor is coarser (we are finer), map to coarse cell.
    if (neighborLevel < currentLevel) {
        int scale = 1 << (currentLevel - neighborLevel);
        // Map ghost coordinate wrapped around into coarse block
        int ghostX = (dx == -1) ? (currentRes/scale - 1) : (dx == 1) ? 0 : (nx / scale);
        int ghostY = (dy == -1) ? (currentRes/scale - 1) : (dy == 1) ? 0 : (ny / scale);
        int ghostZ = (dz == -1) ? (currentRes/scale - 1) : (dz == 1) ? 0 : (nz / scale);
        // Note: neighbor is stored at coarse resolution relative to us
        int nRes = currentRes / scale;
        return p_data[nOffset + cellIndex3D(ghostX, ghostY, ghostZ, nRes)];
    }

    // If neighbor is finer (we are coarser), grab touching fine cell.
    if (neighborLevel > currentLevel) {
        int scale = 1 << (neighborLevel - currentLevel);
        int nRes = currentRes * scale;
        int ghostX = (dx == -1) ? (nRes - 1) : (dx == 1) ? 0 : (nx * scale + scale/2);
        int ghostY = (dy == -1) ? (nRes - 1) : (dy == 1) ? 0 : (ny * scale + scale/2);
        int ghostZ = (dz == -1) ? (nRes - 1) : (dz == 1) ? 0 : (nz * scale + scale/2);
        return p_data[nOffset + cellIndex3D(ghostX, ghostY, ghostZ, nRes)];
    }

    return 0.f;
}

// --- SMOOTHER KERNEL ---

__device__ inline void activateBlock(uint32_t blockIdx, uint32_t* nextList, uint32_t* nextCount, uint32_t* nextStatus) {
    if (atomicExch(&nextStatus[blockIdx], 1) == 0) {
        uint32_t pos = atomicAdd(nextCount, 1);
        nextList[pos] = blockIdx;
    }
}

__global__ void smoothActiveBlocks(
    MSBG grid,
    float* p_data,              // Explicit Solution
    const float* rhs_data,      // Explicit RHS
    const float* betaX, const float* betaY, const float* betaZ, // Explicit Coeffs
    const uint32_t* cellOffsets,
    const uint32_t* fxOffsets, const uint32_t* fyOffsets, const uint32_t* fzOffsets,
    const uint32_t* activeList, uint32_t activeCount,
    uint32_t* nextList, uint32_t* nextCount, uint32_t* nextStatus,
    int rbPhase, float theta, int mgLevel
) {
    uint32_t listIdx = blockIdx.x;
    if (listIdx >= activeCount) return;

    uint32_t gridBlockIdx = activeList[listIdx];
    BlockInfo info = grid.d_blockInfo[gridBlockIdx];

    // Red-Black Check on BLOCK coordinates
    if ((info.coord.x + info.coord.y + info.coord.z) % 2 != rbPhase) return;

    // Calculate resolution for this MG Level
    uint32_t msbgLevel = grid.d_refinementMap[gridBlockIdx];
    int res = c_blockLayouts[msbgLevel].res >> mgLevel;
    if (res < 1) return;

    uint32_t offset = cellOffsets[gridBlockIdx];
    uint32_t fxOff = fxOffsets[gridBlockIdx];
    uint32_t fyOff = fyOffsets[gridBlockIdx];
    uint32_t fzOff = fzOffsets[gridBlockIdx];

    float initialBlockRes = -1.f;
    __shared__ float s_maxRes;

    for (int iter = 0; iter < MAX_BLOCK_ITER; iter++) {
        if (threadIdx.x == 0) s_maxRes = 0.f;
        __syncthreads();

        for (int color = 0; color < 2; color++) {
            int numCells = res * res * res;
            for (int i = threadIdx.x; i < numCells; i += blockDim.x) {
                int z = i / (res * res);
                int rem = i % (res * res);
                int y = rem / res;
                int x = rem % res;

                if ((x + y + z) % 2 != color) continue;

                // Coefficients
                float bW = betaX[fxOff + faceXIndex3D(x, y, z, res)];
                float bE = betaX[fxOff + faceXIndex3D(x+1, y, z, res)];
                float bS = betaY[fyOff + faceYIndex3D(x, y, z, res)];
                float bN = betaY[fyOff + faceYIndex3D(x, y+1, z, res)];
                float bB = betaZ[fzOff + faceZIndex3D(x, y, z, res)];
                float bT = betaZ[fzOff + faceZIndex3D(x, y, z+1, res)];

                float diag = -(bW + bE + bS + bN + bB + bT);
                if (fabsf(diag) < 1e-9f) continue;

                // Neighbors
                float pW = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, -1, 0, 0);
                float pE = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, 1, 0, 0);
                float pS = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, 0, -1, 0);
                float pN = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, 0, 1, 0);
                float pB = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, 0, 0, -1);
                float pT = getPressureLevel(grid, p_data, cellOffsets, gridBlockIdx, make_int3(x, y, z), res, 0, 0, 1);

                float Ax = bW*pW + bE*pE + bS*pS + bN*pN + bB*pB + bT*pT + diag * p_data[offset + i];
                float rhs = rhs_data[offset + i];
                float r = rhs - Ax;

                p_data[offset + i] += OMEGA * (r / diag);

                float absR = fabsf(r);
                atomicMax((unsigned int*)&s_maxRes, __float_as_uint(absR));
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) { if (iter == 0) initialBlockRes = s_maxRes; }
        __syncthreads();
        if (initialBlockRes > 1e-9f && s_maxRes < (initialBlockRes * TARGET_RESIDUAL_REDUCTION)) break;
    }

    // Adaptive Activation Logic
    if (threadIdx.x == 0 && s_maxRes > theta) {
        activateBlock(gridBlockIdx, nextList, nextCount, nextStatus);
        int3 c = info.coord;
        int3 nCoords[6] = {
            make_int3(c.x-1, c.y, c.z), make_int3(c.x+1, c.y, c.z),
            make_int3(c.x, c.y-1, c.z), make_int3(c.x, c.y+1, c.z),
            make_int3(c.x, c.y, c.z-1), make_int3(c.x, c.y, c.z+1)
        };
        for (int k = 0; k < 6; k++) {
            uint32_t nIdx = getNeighborBlockIdx(nCoords[k], grid.indexDims, 0, 0, 0);
            if (nIdx != -1) activateBlock(nIdx, nextList, nextCount, nextStatus);
        }
    }
}

// --- INIT KERNEL ---
__global__ inline void initActiveBlocks(
    MSBG grid, uint32_t* activeList, uint32_t* activeCount, uint32_t* blockStatus,
    const uint32_t targetLevel, int mgLevel
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid.numBlocks) return;

    // Check if block exists at this MG level
    uint32_t msbgLevel = grid.d_refinementMap[idx];
    int res = c_blockLayouts[msbgLevel].res >> mgLevel;

    if (res >= 1) { // Block active at this level
        const uint32_t pos = atomicAdd(activeCount, 1);
        activeList[pos] = idx;
        blockStatus[idx] = 1;
    } else {
        blockStatus[idx] = 0;
    }
}

// --- CLASS ---

class AdaptiveSmoother {
public:
    uint32_t* d_activeList[2];
    uint32_t* d_activeCount[2];
    uint32_t* d_blockStatus[2];
    uint32_t numBlocks;

    AdaptiveSmoother(uint32_t n) : numBlocks(n) {
        cudaMalloc(&d_activeList[0], numBlocks * sizeof(uint32_t));
        cudaMalloc(&d_activeList[1], numBlocks * sizeof(uint32_t));
        cudaMalloc(&d_activeCount[0], sizeof(uint32_t));
        cudaMalloc(&d_activeCount[1], sizeof(uint32_t));
        cudaMalloc(&d_blockStatus[0], numBlocks * sizeof(uint32_t));
        cudaMalloc(&d_blockStatus[1], numBlocks * sizeof(uint32_t));
    }

    ~AdaptiveSmoother() {
        cudaFree(d_activeList[0]); cudaFree(d_activeList[1]);
        cudaFree(d_activeCount[0]); cudaFree(d_activeCount[1]);
        cudaFree(d_blockStatus[0]); cudaFree(d_blockStatus[1]);
    }

    void solveLevel(
        MSBGManager& mg, int mgLevel,
        float* x, float* b,
        float* bx, float* by, float* bz,
        uint32_t* c_off, uint32_t* fx_off, uint32_t* fy_off, uint32_t* fz_off,
        int iterations
    ) {
        int curr = 0; int next = 1;
        cudaMemset(d_activeCount[curr], 0, sizeof(uint32_t));
        cudaMemset(d_blockStatus[curr], 0, numBlocks * sizeof(uint32_t));

        int threads = 256;
        int blocks = (numBlocks + threads - 1) / threads;

        initActiveBlocks<<<blocks, threads>>>(
            mg.grid, d_activeList[curr], d_activeCount[curr], d_blockStatus[curr], MAX_LEVELS-1, mgLevel
        );
        cudaDeviceSynchronize();

        for (int iter = 0; iter < iterations; iter++) {
            uint32_t h_count = 0;
            cudaMemcpy(&h_count, d_activeCount[curr], sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (h_count == 0) break;

            thrust::device_ptr<uint32_t> ptr(d_activeList[curr]);
            MortonCompare comp(mg.grid.d_blockInfo);
            thrust::sort(thrust::device, ptr, ptr + h_count, comp);

            cudaMemset(d_activeCount[next], 0, sizeof(uint32_t));
            cudaMemset(d_blockStatus[next], 0, numBlocks * sizeof(uint32_t));

            for (int rb = 0; rb < 2; rb++) {
                smoothActiveBlocks<<<h_count, 256>>>(
                    mg.grid, x, b, bx, by, bz,
                    c_off, fx_off, fy_off, fz_off,
                    d_activeList[curr], h_count,
                    d_activeList[next], d_activeCount[next], d_blockStatus[next],
                    rb, 1e-4f, mgLevel
                );
            }
            std::swap(curr, next);
        }
    }
};

#endif //ANITOWAVE_ADAPTIVE_SMOOTHER_CUH