#ifndef ANITOWAVE_ADAPTIVE_MULTIGRID_CUH
#define ANITOWAVE_ADAPTIVE_MULTIGRID_CUH

#define BLOCK_BASE_RES 16
#define MAX_LEVELS 3
#define WARP_SIZE 32

#define DELTA_G 0.75f
#define SIGMA_CF (1 / (4 * DELTA_G))
#define SIGMA_FC (2 * SIGMA_CF)

enum DataChannel {
    // Velocity components (MAC grid - face-centered)
    VELOCITY_X = 0,
    VELOCITY_Y = 1,
    VELOCITY_Z = 2,

    // Pressure (cell-centered)
    PRESSURE = 3,

    // Phase field φ ∈ [0,1] (face-centered, computed from mass)
    PHASE_FIELD_X = 4,
    PHASE_FIELD_Y = 5,
    PHASE_FIELD_Z = 6,

    // Face coefficients β = 1/ρ for Poisson equation (face-centered)
    BETA_COEFF_X = 7,
    BETA_COEFF_Y = 8,
    BETA_COEFF_Z = 9,

    // Raw mass density from particles (face-centered)
    MASS_DENSITY_X = 10,
    MASS_DENSITY_Y = 11,
    MASS_DENSITY_Z = 12,

    // Momentum from particles (face-centered) - temporary for P2G
    MOMENTUM_X = 13,
    MOMENTUM_Y = 14,
    MOMENTUM_Z = 15,

    // Auxiliary fields
    DIVERGENCE = 16,        // Cell-centered, for pressure solve
    DISTANCE_FIELD = 17,    // Cell-centered, for refinement
    VORTICITY = 18,         // Cell-centered, optional refinement criterion

    // Cell flags
    CELL_TYPE = 19,         // Cell-centered: air/liquid/solid

    NUM_CHANNELS = 20
};

enum Axis { AXIS_X = 0, AXIS_Y = 1, AXIS_Z = 2 };

struct BlockInfo {
    int3 coord;
    uint32_t level;
    uint32_t morton;
};

struct BlockLayout {
    uint32_t numCells;
    uint32_t numFacesX;
    uint32_t numFacesY;
    uint32_t numFacesZ;
    uint32_t totalSize;
    int res;
};

__constant__ inline BlockLayout c_blockLayouts[MAX_LEVELS];

struct MSBG {
    // Block topology
    int3 indexDims;
    uint32_t numBlocks;
    uint32_t numActiveBlocks;

    // Refinement level for each block
    uint32_t* d_refinementMap;

    // Block metadata
    BlockInfo* d_blockInfo;

    // Data offsets
    // in SoA
    // Offset into array for each channel type and block
    uint32_t* d_cellOffsets;
    uint32_t* d_faceXOffsets;
    uint32_t* d_faceYOffsets;
    uint32_t* d_faceZOffsets;

    // Grid data
    float** d_cellData;
    float** d_faceXData;
    float** d_faceYData;
    float** d_faceZData;

    // Total sizes
    uint32_t totalCells;
    uint32_t totalFacesX;
    uint32_t totalFacesY;
    uint32_t totalFacesZ;

    // Parameters
    float blockSize;
    float3 domainMin;
};

//// Utils


// Compute block layout for a given level
__host__ inline BlockLayout computeBlockLayout(uint8_t level) {
    BlockLayout layout{};
    layout.res = BLOCK_BASE_RES >> level;
    int r = layout.res;

    layout.numCells = r * r * r;
    layout.numFacesX = (r + 1) * r * r;
    layout.numFacesY = r * (r + 1) * r;
    layout.numFacesZ = r * r * (r + 1);
    layout.totalSize = layout.numCells + layout.numFacesX + layout.numFacesY + layout.numFacesZ;

    return layout;
}

// Initialize GPU memory with block layouts
__host__ void initializeBlockLayouts() {
    BlockLayout layouts[MAX_LEVELS];
    for (int i = 0; i < MAX_LEVELS; i++) {
        layouts[i] = computeBlockLayout(i);
    }
    cudaError_t err = cudaMemcpyToSymbol(c_blockLayouts, layouts, MAX_LEVELS * sizeof(BlockLayout));
    if (err != cudaSuccess) {
        printf("Error in initializeBlockLayouts: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpyToSymbol(c_blockLayouts, layouts, MAX_LEVELS * sizeof(BlockLayout));
}

__device__ __host__ inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

__device__ __host__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

__device__ __host__ inline int3 worldToBlockCoord(float3 pos, float blockSize, float3 domainMin) {
    float3 rel = make_float3(pos.x - domainMin.x, pos.y - domainMin.y, pos.z - domainMin.z);

    return make_int3(
        (int)floorf(rel.x / blockSize),
        (int)floorf(rel.y / blockSize),
        (int)floorf(rel.z / blockSize)
        );
}

__device__ __host__ inline uint32_t blockCoordToIndex(int3 coord, int3 dims) {
    return coord.x + dims.x * (coord.y + dims.y * coord.z);
}

__device__ inline uint32_t cellIndex3D(int x, int y, int z, int res) {
    return x + res * (y + res * z);
}

__device__ inline uint32_t faceXIndex3D(int x, int y, int z, int res) {
    return x + (res + 1) * (y + res * z);
}

__device__ inline uint32_t faceYIndex3D(int x, int y, int z, int res) {
    return x + res * (y + (res + 1) * z);
}

__device__ inline uint32_t faceZIndex3D(int x, int y, int z, int res) {
    return x + res * (y + res * z);
}

__device__ inline uint32_t getNeighborBlockIdx(int3 blockCoord, int3 dims, int dx, int dy, int dz) {
    int3 nCoord = make_int3(blockCoord.x + dx, blockCoord.y + dy, blockCoord.z + dz);

    if (nCoord.x < 0 || nCoord.x >= dims.x ||
        nCoord.y < 0 || nCoord.y >= dims.y ||
        nCoord.z < 0 || nCoord.z >= dims.z) {
        return -1;
        }

    return blockCoordToIndex(nCoord, dims);
}

//// Grid access

__device__ inline float& accessCell(
    MSBG& grid,
    DataChannel channel,
    uint32_t blockId,
    int x, int y, int z,
    uint8_t level
) {
    uint32_t offset = grid.d_cellOffsets[blockId];
    int res = c_blockLayouts[level].res;
    uint32_t idx = cellIndex3D(x, y, z, res);
    return grid.d_cellData[channel][offset + idx];
}

__device__ inline float& accessFaceX(
    MSBG& grid,
    DataChannel channel,
    uint32_t blockId,
    int x, int y, int z,
    uint8_t level
) {
    uint32_t offset = grid.d_faceXOffsets[blockId];
    int res = c_blockLayouts[level].res;
    uint32_t idx = faceXIndex3D(x, y, z, res);
    return grid.d_faceXData[channel][offset + idx];
}

__device__ inline float& accessFaceY(
    MSBG& grid,
    DataChannel channel,
    uint32_t blockId,
    int x, int y, int z,
    uint8_t level
) {
    uint32_t offset = grid.d_faceYOffsets[blockId];
    int res = c_blockLayouts[level].res;
    uint32_t idx = faceYIndex3D(x, y, z, res);
    return grid.d_faceYData[channel][offset + idx];
}

__device__ inline float& accessFaceZ(
    MSBG& grid,
    DataChannel channel,
    uint32_t blockId,
    int x, int y, int z,
    uint8_t level
) {
    uint32_t offset = grid.d_faceZOffsets[blockId];
    int res = c_blockLayouts[level].res;
    uint32_t idx = faceZIndex3D(x, y, z, res);
    return grid.d_faceZData[channel][offset + idx];
}

//// Block management

__global__ void initRefinementMap(uint32_t* map, uint32_t value, uint32_t N)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    map[idx] = value;
}

__global__ void initBlockInfo(BlockInfo* info, int3 dims) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t numBlocks = dims.x * dims.y * dims.z;
    if (idx >= numBlocks) return;

    int z = idx / (dims.x * dims.y);
    int rem = idx % (dims.x * dims.y);
    int y = rem / dims.x;
    int x = rem % dims.x;

    info[idx].coord = make_int3(x, y, z);
    info[idx].morton = morton3D(x, y, z);
    info[idx].level = 0;
}
// Mark active blocks from particle positions
__global__ void markActiveBlocks(
    const float3* particles,
    const float* particleRadii,
    uint32_t numParticles,
    uint32_t* refinementMap,
    int3 indexDims,
    float blockSize,
    float3 domainMin
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = particles[idx];
    float radius = particleRadii[idx];

    // r_p = r_0 / 2^l_p
    // l_p = log2(r_0 / r_p)
    float r0 = blockSize / BLOCK_BASE_RES;
    int plev = static_cast<int>(floorf(max(0.0f, log2f(r0 / radius))));
    plev = max(0, min(plev, MAX_LEVELS - 1));
    uint8_t particleLevel = static_cast<uint8_t>(plev);

    int3 bCoord = worldToBlockCoord(pos, blockSize, domainMin);

    // mark block containing particle and its neighbors
    int res = c_blockLayouts[particleLevel].res;
    float cellSize = blockSize / float(res);            // world-size of a cell at particleLevel
    int support = max(0, static_cast<int>(ceilf(radius * cellSize / blockSize)));

    for (int dz = -support; dz <= support; dz++) {
        for (int dy = -support; dy <= support; dy++) {
            for (int dx = -support; dx <= support; dx++) {
                int3 nCoord = make_int3(bCoord.x + dx, bCoord.y + dy, bCoord.z + dz);

                if (nCoord.x < 0 || nCoord.x >= indexDims.x ||
                    nCoord.y < 0 || nCoord.y >= indexDims.y ||
                    nCoord.z < 0 || nCoord.z >= indexDims.z) continue;

                uint32_t blockIdx = blockCoordToIndex(nCoord, indexDims);
                atomicMin(&refinementMap[blockIdx], particleLevel);
            }
        }
    }
}

__global__ void enforceGrading(
    uint32_t* refinementMap,
    int3 dims
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dims.x || y >= dims.y || z >= dims.z) return;

    uint32_t idx = blockCoordToIndex(make_int3(x, y, z), dims);
    uint32_t level = refinementMap[idx];

    if (level == MAX_LEVELS) return;

    // Check all neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || nx >= dims.x ||
                    ny < 0 || ny >= dims.y ||
                    nz < 0 || nz >= dims.z) continue;

                uint32_t nIdx = blockCoordToIndex(make_int3(nx, ny, nz), dims);
                uint32_t nLevel = refinementMap[nIdx];

                if (nLevel != MAX_LEVELS && abs(static_cast<int>(level) - static_cast<int>(nLevel)) > 1) {
                    atomicMin(&refinementMap[idx], nLevel + 1u);
                }
            }
        }
    }
}

__global__ void countBlockElements(
    const uint32_t* refinementMap,
    uint32_t* cellCounts,
    uint32_t* faceXCounts,
    uint32_t* faceYCounts,
    uint32_t* faceZCounts,
    uint32_t numBlocks
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    if (uint32_t level = refinementMap[idx]; level == MAX_LEVELS) {
        cellCounts[idx] = 0;
        faceXCounts[idx] = 0;
        faceYCounts[idx] = 0;
        faceZCounts[idx] = 0;
    } else {
        const BlockLayout layout = c_blockLayouts[level];
        cellCounts[idx] = layout.numCells;
        faceXCounts[idx] = layout.numFacesX;
        faceYCounts[idx] = layout.numFacesY;
        faceZCounts[idx] = layout.numFacesZ;
    }
}

__device__ inline float getScalingFactor(uint32_t currentLevel, uint32_t neighborBlockIdx, const uint32_t* refinementMap) {
    uint32_t neighborLevel = (neighborBlockIdx != -1) ? refinementMap[neighborBlockIdx] : currentLevel;
    int levelDiff = currentLevel - neighborLevel;


    // case 0: s <- 2^l
    // case 1: s <- 2^l * sigma_CF
    // case -1: s <- 2^l * sigma_FC

    auto s = (float)(1 << currentLevel);

    switch (levelDiff) {
        case 0: break;
        case 1: s *= SIGMA_CF; break;
        case -1: s *= SIGMA_FC; break;
        default: break;
    }

    return s;
}

__global__ void galerkinCoarsen(MSBG grid, const float* __restrict__ d_fineCoeffs, float* __restrict__ d_coarseCoeffs, int axis, uint32_t coarseLevel) {
    uint32_t idx = blockIdx.x;
    if (idx >= grid.numBlocks) return;

    uint32_t nativeLevel = grid.d_refinementMap[idx];

    if (nativeLevel >= coarseLevel)
    {
        if (nativeLevel == coarseLevel)
        {
            BlockLayout layout = c_blockLayouts[coarseLevel];
            int res = layout.res;

            int dimX = res + (axis == AXIS_X ? 1 : 0);
            int dimY = res + (axis == AXIS_Y ? 1 : 0);
            int dimZ = res + (axis == AXIS_Z ? 1 : 0);

            int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
            if (tx >= dimX || ty >= dimY || tz >= dimZ) return;

            uint32_t offset = (axis == AXIS_X) ? grid.d_faceXOffsets[idx]
                            : (axis == AXIS_Y) ? grid.d_faceYOffsets[idx]
                            :                    grid.d_faceZOffsets[idx];

            uint32_t faceIdx;
            if (axis == AXIS_X) faceIdx = faceXIndex3D(tx, ty, tz, res);
            else if (axis == AXIS_Y) faceIdx = faceYIndex3D(tx, ty, tz, res);
            else faceIdx = faceZIndex3D(tx, ty, tz, res);

            d_coarseCoeffs[offset + faceIdx] = d_fineCoeffs[offset + faceIdx];
        }
        return;
    }

    if (nativeLevel != (coarseLevel - 1)) return;

    BlockLayout coarseLayout = c_blockLayouts[coarseLevel];
    int coarseRes = coarseLayout.res;

    BlockLayout fineLayout = c_blockLayouts[coarseLevel - 1];
    int fineRes = fineLayout.res;

    // One thread per face within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int dimX = coarseRes + (axis == AXIS_X ? 1 : 0);
    int dimY = coarseRes + (axis == AXIS_Y ? 1 : 0);
    int dimZ = coarseRes + (axis == AXIS_Z ? 1 : 0);

    if (tx >= dimX || ty >= dimY || tz >= dimZ) return;

    uint32_t neighborIdx = -1;
    bool isBoundary = false;
    int3 bCoord = grid.d_blockInfo[idx].coord;

    if (axis == AXIS_X) {
        if (tx == 0) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, -1, 0, 0); isBoundary = true; }
        if (tx == coarseRes) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, 1, 0, 0); isBoundary = true; }
    } else if (axis == AXIS_Y) {
        if (ty == 0) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, 0, -1, 0); isBoundary = true; }
        if (ty == coarseRes) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, 0, 1, 0); isBoundary = true; }
    } else {
        if (tz == 0) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, 0, 0, -1); isBoundary = true; }
        if (tz == coarseRes) { neighborIdx = getNeighborBlockIdx(bCoord,grid.indexDims, 0, 0, -1); isBoundary = true; }
    }

    float s;
    if (isBoundary) {
        s = getScalingFactor(coarseLevel, neighborIdx, grid.d_refinementMap);
    } else {
        s = (float)(1 << coarseLevel);
    }

    float sum = 0.f;
    uint32_t fineBlockOffset = (axis == AXIS_X) ? grid.d_faceXOffsets[idx]
                             : (axis == AXIS_Y) ? grid.d_faceYOffsets[idx]
                             :                    grid.d_faceZOffsets[idx];

    const int fx_start = tx * 2;
    const int fy_start = ty * 2;
    const int fz_start = tz * 2;

    for (int d1 = 0; d1 < 2; d1++) {
        for (int d2 = 0; d2 < 2; d2++) {
            int f_x = fx_start;
            int f_y = fy_start;
            int f_z = fz_start;

            if (axis == AXIS_X) {
                f_y += d1;
                f_z += d2;
            } else if (axis == AXIS_Y) {
                f_x += d1;
                f_z += d2;
            } else {
                f_x += d1;
                f_y += d2;
            }

            uint32_t fineIdx;
            if (axis == AXIS_X) fineIdx = faceXIndex3D(f_x, f_y, f_z, fineRes);
            else if (axis == AXIS_Y) fineIdx = faceYIndex3D(f_x, f_y, f_z, fineRes);
            else fineIdx = faceZIndex3D(f_x, f_y, f_z, fineRes);

            sum += d_fineCoeffs[fineBlockOffset + fineIdx];
        }
    }

    uint32_t coarseBlockOffset = (axis == AXIS_X) ? grid.d_faceXOffsets[idx]
                                : (axis == AXIS_Y) ? grid.d_faceYOffsets[idx]
                                : grid.d_faceZOffsets[idx];

    uint32_t coarseIdx;
    if (axis == AXIS_X) coarseIdx = faceXIndex3D(tx, ty, tz, coarseRes);
    else if (axis == AXIS_Y) coarseIdx = faceYIndex3D(tx, ty, tz, coarseRes);
    else coarseIdx = faceZIndex3D(tx, ty, tz, coarseRes);

    d_coarseCoeffs[coarseBlockOffset + coarseIdx] = s * (sum * 0.25f);
}

//// Grid Manager class

class MSBGManager {
public:
    MSBG grid;

    MSBGManager(int3 dims, float blockSize, float3 domainMin) {
        grid.indexDims = dims;
        grid.blockSize = blockSize;
        grid.domainMin = domainMin;
        grid.numBlocks = dims.x * dims.y * dims.z;

        // Allocate topology
        cudaMalloc(&grid.d_refinementMap, grid.numBlocks * sizeof(uint32_t));
        cudaMemset(grid.d_refinementMap, MAX_LEVELS - 1, grid.numBlocks * sizeof(uint32_t));

        cudaMalloc(&grid.d_blockInfo, grid.numBlocks * sizeof(BlockInfo));

        int threads = 256;
        int blocks = (grid.numBlocks + threads - 1) / threads;
        initBlockInfo<<<blocks, threads>>>(grid.d_blockInfo, dims);
        cudaDeviceSynchronize();

        cudaMalloc(&grid.d_cellOffsets, grid.numBlocks * sizeof(uint32_t));
        cudaMalloc(&grid.d_faceXOffsets, grid.numBlocks * sizeof(uint32_t));
        cudaMalloc(&grid.d_faceYOffsets, grid.numBlocks * sizeof(uint32_t));
        cudaMalloc(&grid.d_faceZOffsets, grid.numBlocks * sizeof(uint32_t));

        // Initialize to 0
        cudaMemset(grid.d_cellOffsets, 0, grid.numBlocks * sizeof(uint32_t));
        cudaMemset(grid.d_faceXOffsets, 0, grid.numBlocks * sizeof(uint32_t));
        cudaMemset(grid.d_faceYOffsets, 0, grid.numBlocks * sizeof(uint32_t));
        cudaMemset(grid.d_faceZOffsets, 0, grid.numBlocks * sizeof(uint32_t));

        // Init block layouts in constant GPU mem
        initializeBlockLayouts();
        grid.d_cellData = nullptr;
        grid.d_faceXData = nullptr;
        grid.d_faceYData = nullptr;
        grid.d_faceZData = nullptr;
        grid.totalCells = 0;
        grid.numActiveBlocks = 0;
    }

    ~MSBGManager() {
        cleanup();
    }

    void cleanup() {
        if (grid.d_refinementMap) cudaFree(grid.d_refinementMap);
        if (grid.d_blockInfo) cudaFree(grid.d_blockInfo);
        if (grid.d_cellOffsets) cudaFree(grid.d_cellOffsets);
        if (grid.d_faceXOffsets) cudaFree(grid.d_faceXOffsets);
        if (grid.d_faceYOffsets) cudaFree(grid.d_faceYOffsets);
        if (grid.d_faceZOffsets) cudaFree(grid.d_faceZOffsets);

        float* h_temp[NUM_CHANNELS] = {};

        // free data channels
        if (grid.d_cellData) {
            cudaMemcpy(h_temp, grid.d_cellData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
            for (auto & i : h_temp) {
                if (i) cudaFree(i); i = nullptr;
            }
            cudaFree(grid.d_cellData);
            grid.d_cellData = nullptr;
        }

        memset(h_temp, 0, NUM_CHANNELS * sizeof(float*));
        if (grid.d_faceXData) {
            cudaMemcpy(h_temp, grid.d_faceXData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
            for (auto & i : h_temp) {
                if (i) cudaFree(i);
            }
            cudaFree(grid.d_faceXData);
            grid.d_faceXData = nullptr;
        }

        memset(h_temp, 0, NUM_CHANNELS * sizeof(float*));
        if (grid.d_faceYData) {
            cudaMemcpy(h_temp, grid.d_faceYData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
            for (auto & i : h_temp) {
                if (i) cudaFree(i);
            }
            cudaFree(grid.d_faceYData);
            grid.d_faceYData = nullptr;
        }

        memset(h_temp, 0, NUM_CHANNELS * sizeof(float*));
        if (grid.d_faceZData) {
            cudaMemcpy(h_temp, grid.d_faceZData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
            for (auto & i : h_temp) {
                if (i) cudaFree(i);
            }
            cudaFree(grid.d_faceZData);
            grid.d_faceZData = nullptr;
        }
    }

    void allocateChannels() {
        auto** h_cellData = new float*[NUM_CHANNELS];
        auto** h_faceXData = new float*[NUM_CHANNELS];
        auto** h_faceYData = new float*[NUM_CHANNELS];
        auto** h_faceZData = new float*[NUM_CHANNELS];

        memset(h_cellData, 0, NUM_CHANNELS * sizeof(float*));
        memset(h_faceXData, 0, NUM_CHANNELS * sizeof(float*));
        memset(h_faceYData, 0, NUM_CHANNELS * sizeof(float*));
        memset(h_faceZData, 0, NUM_CHANNELS * sizeof(float*));

        // Allocate channels that are cell-centered
        if (grid.totalCells > 0) {
            cudaMalloc(&h_cellData[PRESSURE], grid.totalCells * sizeof(float));
            cudaMalloc(&h_cellData[DIVERGENCE], grid.totalCells * sizeof(float));
            cudaMalloc(&h_cellData[DISTANCE_FIELD], grid.totalCells * sizeof(float));
            cudaMalloc(&h_cellData[VORTICITY], grid.totalCells * sizeof(float));
            cudaMalloc(&h_cellData[CELL_TYPE], grid.totalCells * sizeof(float));
        }

        // Allocate face-centered channels
        if (grid.totalFacesX > 0) {
            cudaMalloc(&h_faceXData[VELOCITY_X], grid.totalFacesX * sizeof(float));
            cudaMalloc(&h_faceXData[PHASE_FIELD_X], grid.totalFacesX * sizeof(float));
            cudaMalloc(&h_faceXData[BETA_COEFF_X], grid.totalFacesX * sizeof(float));
            cudaMalloc(&h_faceXData[MASS_DENSITY_X], grid.totalFacesX * sizeof(float));
            cudaMalloc(&h_faceXData[MOMENTUM_X], grid.totalFacesX * sizeof(float));
        }

        if (grid.totalFacesY > 0) {
            cudaMalloc(&h_faceYData[VELOCITY_Y], grid.totalFacesY * sizeof(float));
            cudaMalloc(&h_faceYData[PHASE_FIELD_Y], grid.totalFacesY * sizeof(float));
            cudaMalloc(&h_faceYData[BETA_COEFF_Y], grid.totalFacesY * sizeof(float));
            cudaMalloc(&h_faceYData[MASS_DENSITY_Y], grid.totalFacesY * sizeof(float));
            cudaMalloc(&h_faceYData[MOMENTUM_Y], grid.totalFacesY * sizeof(float));
        }

        if (grid.totalFacesZ > 0) {
            cudaMalloc(&h_faceZData[VELOCITY_Z], grid.totalFacesZ * sizeof(float));
            cudaMalloc(&h_faceZData[PHASE_FIELD_Z], grid.totalFacesZ * sizeof(float));
            cudaMalloc(&h_faceZData[BETA_COEFF_Z], grid.totalFacesZ * sizeof(float));
            cudaMalloc(&h_faceZData[MASS_DENSITY_Z], grid.totalFacesZ * sizeof(float));
            cudaMalloc(&h_faceZData[MOMENTUM_Z], grid.totalFacesZ * sizeof(float));
        }

        // Copy pointer arrays to device
        cudaMalloc(&grid.d_cellData, NUM_CHANNELS * sizeof(float*));
        cudaMalloc(&grid.d_faceXData, NUM_CHANNELS * sizeof(float*));
        cudaMalloc(&grid.d_faceYData, NUM_CHANNELS * sizeof(float*));
        cudaMalloc(&grid.d_faceZData, NUM_CHANNELS * sizeof(float*));

        cudaMemcpy(grid.d_cellData, h_cellData, NUM_CHANNELS * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(grid.d_faceXData, h_faceXData, NUM_CHANNELS * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(grid.d_faceYData, h_faceYData, NUM_CHANNELS * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(grid.d_faceZData, h_faceZData, NUM_CHANNELS * sizeof(float*), cudaMemcpyHostToDevice);

        delete[] h_cellData;
        delete[] h_faceXData;
        delete[] h_faceYData;
        delete[] h_faceZData;
    }

    void runGalerkin(const float* d_fineBetaX, const float* d_fineBetaY, const float* d_fineBetaZ,
                     float* d_coarseBetaX, float* d_coarseBetaY, float* d_coarseBetaZ,
                     const uint32_t targetCoarseLevel) const {
        if (targetCoarseLevel == 0 || targetCoarseLevel >= MAX_LEVELS) return;

        int dim = c_blockLayouts[targetCoarseLevel].res + 1;
        dim3 threads(dim, dim, dim);
        dim3 blocks(this->grid.numBlocks);

        galerkinCoarsen<<<blocks, threads>>>(this->grid, d_fineBetaX, d_coarseBetaX, AXIS_X, targetCoarseLevel);
        galerkinCoarsen<<<blocks, threads>>>(this->grid, d_fineBetaY, d_coarseBetaY, AXIS_Y, targetCoarseLevel);
        galerkinCoarsen<<<blocks, threads>>>(this->grid, d_fineBetaZ, d_coarseBetaZ, AXIS_Z, targetCoarseLevel);

        cudaDeviceSynchronize();
    }
};


void computeCoarseOffsets(
    MSBGManager& mg,
    int mgLevel,
    uint32_t* d_offsets_out,
    uint32_t& totalSize_out
    )
{
    std::vector<uint32_t>  h_refinement(mg.grid.numBlocks);
    cudaMemcpy(h_refinement.data(), mg.grid.d_refinementMap, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<uint32_t> h_counts(mg.grid.numBlocks);
    for (size_t i = 0; i < mg.grid.numBlocks; i++)
    {
        uint32_t msbgLevel = h_refinement[i];
        int res = c_blockLayouts[msbgLevel].res >> mgLevel;
        if (res < 1) res = 0;
        h_counts[i] = res * res * res;
    }

    std::vector<uint32_t> h_offsets(mg.grid.numBlocks);
    uint32_t total = 0;
    for (size_t i = 0; i < mg.grid.numBlocks; i++)
    {
        h_offsets[i] = total;
        total += h_counts[i];
    }
    totalSize_out = total;

    cudaMemcpy(d_offsets_out, h_offsets.data(), mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
}
#endif //ANITOWAVE_ADAPTIVE_MULTIGRID_CUH