#ifndef ANITOWAVE_ADAPTIVE_MULTIGRID_CUH
#define ANITOWAVE_ADAPTIVE_MULTIGRID_CUH

#define BLOCK_BASE_RES 16
#define MAX_LEVELS 3
#define WARP_SIZE 32

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

struct Multigrid {
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

//// Grid access

__device__ inline float& accessCell(
    Multigrid& grid,
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
    Multigrid& grid,
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
    Multigrid& grid,
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
    Multigrid& grid,
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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint32_t level = refinementMap[idx];

    if (level == MAX_LEVELS) {
        cellCounts[idx] = 0;
        faceXCounts[idx] = 0;
        faceYCounts[idx] = 0;
        faceZCounts[idx] = 0;
    } else {
        BlockLayout layout = c_blockLayouts[level];
        cellCounts[idx] = layout.numCells;
        faceXCounts[idx] = layout.numFacesX;
        faceYCounts[idx] = layout.numFacesY;
        faceZCounts[idx] = layout.numFacesZ;
    }
}

//// Grid Manager class

class MultigridManager {
public:
    Multigrid grid;

    MultigridManager(int3 dims, float blockSize, float3 domainMin) {
        grid.indexDims = dims;
        grid.blockSize = blockSize;
        grid.domainMin = domainMin;
        grid.numBlocks = dims.x * dims.y * dims.z;

        // Allocate topology
        cudaMalloc(&grid.d_refinementMap, grid.numBlocks * sizeof(uint32_t));
        cudaMemset(grid.d_refinementMap, MAX_LEVELS - 1, grid.numBlocks * sizeof(uint32_t));

        // Init block layouts in constant GPU mem
        initializeBlockLayouts();

        grid.d_cellOffsets = nullptr;
        grid.d_faceXOffsets = nullptr;
        grid.d_faceYOffsets = nullptr;
        grid.d_faceZOffsets = nullptr;
        grid.d_cellData = nullptr;
        grid.d_faceXData = nullptr;
        grid.d_faceYData = nullptr;
        grid.d_faceZData = nullptr;
        grid.totalCells = 0;
        grid.numActiveBlocks = 0;
    }

    ~MultigridManager() {
        cleanup();
    }

    void cleanup() {
        if (grid.d_refinementMap) cudaFree(grid.d_refinementMap);
        if (grid.d_blockInfo) cudaFree(grid.d_blockInfo);
        if (grid.d_cellOffsets) cudaFree(grid.d_cellOffsets);
        if (grid.d_faceXOffsets) cudaFree(grid.d_faceXOffsets);
        if (grid.d_faceYOffsets) cudaFree(grid.d_faceYOffsets);
        if (grid.d_faceZOffsets) cudaFree(grid.d_faceZOffsets);

        // free data channels
        if (grid.d_cellData) {
            for (int i = 0; i < NUM_CHANNELS; i++) {
                if (grid.d_cellData[i]) cudaFree(grid.d_cellData[i]);
            }
            cudaFree(grid.d_cellData);
        }

        if (grid.d_faceXData) {
            for (int i = 0; i < NUM_CHANNELS; i++) {
                if (grid.d_faceXData[i]) cudaFree(grid.d_faceXData[i]);
            }
            cudaFree(grid.d_faceXData);
        }

        if (grid.d_faceYData) {
            for (int i = 0; i < NUM_CHANNELS; i++) {
                if (grid.d_faceYData[i]) cudaFree(grid.d_faceYData[i]);
            }
            cudaFree(grid.d_faceYData);
        }

        if (grid.d_faceZData) {
            for (int i = 0; i < NUM_CHANNELS; i++) {
                if (grid.d_faceZData[i]) cudaFree(grid.d_faceZData[i]);
            }
            cudaFree(grid.d_faceZData);
        }
    }

    void allocateChannels() {
        float** h_cellData = new float*[NUM_CHANNELS];
        float** h_faceXData = new float*[NUM_CHANNELS];
        float** h_faceYData = new float*[NUM_CHANNELS];
        float** h_faceZData = new float*[NUM_CHANNELS];

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
};
#endif //ANITOWAVE_ADAPTIVE_MULTIGRID_CUH