#ifndef ANITOWAVE_ADAPTIVE_P2G_CUH
#define ANITOWAVE_ADAPTIVE_P2G_CUH

#include <MSBG.cuh>

// w(r) = (max(0, 1 - (r/R)^2))^3
__device__ inline float computeKernelWeight(float distSq, float radiusSq)
{
    if (distSq <= radiusSq) return 0.f;
    const float x = 1.f - (distSq / radiusSq);
    return x * x * x;
}

// --- P2G Kernel ---
// Transfers Mass and Momentum from Particles to Grid
__global__ void p2g(
    MSBG grid,
    const float3* __restrict__ p_pos,
    const float3* __restrict__ p_vel,
    uint32_t numParticles,
    float particleMass, // Simplified: All particles have same mass for now
    float particleRadius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = p_pos[idx];
    float3 vel = p_vel[idx];

    // 1. Find Block
    int3 bCoord = worldToBlockCoord(pos, grid.blockSize, grid.domainMin);

    // Bounds check
    if (bCoord.x < 0 || bCoord.x >= grid.indexDims.x ||
        bCoord.y < 0 || bCoord.y >= grid.indexDims.y ||
        bCoord.z < 0 || bCoord.z >= grid.indexDims.z) return;

    uint32_t blockIdx = blockCoordToIndex(bCoord, grid.indexDims);
    uint32_t level = grid.d_refinementMap[blockIdx]; // Native level

    if (level >= MAX_LEVELS) return;

    // 2. Determine Kernel Support
    // The paper uses a support radius equal to the grid spacing of the particle's level.
    // For simplicity in this kernel, we iterate over the 3x3x3 neighborhood of cells
    // around the particle to accumulate weights.

    int res = c_blockLayouts[level].res;
    float cellSize = grid.blockSize / (float)res;
    float supportRadius = cellSize; // r_p = dx (Section 3.3)
    float supportRadiusSq = supportRadius * supportRadius;

    // Local position within block
    float3 blockOrigin;
    blockOrigin.x = bCoord.x * grid.blockSize;
    blockOrigin.y = bCoord.y * grid.blockSize;
    blockOrigin.z = bCoord.z * grid.blockSize;
    float3 localPos = make_float3(pos.x - blockOrigin.x, pos.y - blockOrigin.y, pos.z - blockOrigin.z);

    // Grid index of the cell containing the particle
    int cx = (int)(localPos.x / cellSize);
    int cy = (int)(localPos.y / cellSize);
    int cz = (int)(localPos.z / cellSize);

    // Iterate over 3x3x3 neighbor cells to splat
    // (In a full production implementation, we would use the 8-color scheme for performance)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;

                // Clamp to block bounds (Simplified: ignoring cross-block splatting for this snippet)
                // Real implementation needs getNeighborBlockIdx for boundary cells
                if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res) continue;

                uint32_t offsetCell = grid.d_cellOffsets[blockIdx];
                uint32_t offsetFx = grid.d_faceXOffsets[blockIdx];
                uint32_t offsetFy = grid.d_faceYOffsets[blockIdx];
                uint32_t offsetFz = grid.d_faceZOffsets[blockIdx];

                // Cell Center Position (for Mass)
                float3 cellCenter = make_float3((nx + 0.5f) * cellSize, (ny + 0.5f) * cellSize, (nz + 0.5f) * cellSize);
                float distSq = (localPos.x - cellCenter.x)*(localPos.x - cellCenter.x) +
                               (localPos.y - cellCenter.y)*(localPos.y - cellCenter.y) +
                               (localPos.z - cellCenter.z)*(localPos.z - cellCenter.z);

                float w = computeKernelWeight(distSq, supportRadiusSq);
                if (w > 0.0f) {
                    // Splat Mass (Divergence channel acts as RHS density for now)
                    atomicAdd(&grid.d_cellData[DIVERGENCE][offsetCell + cellIndex3D(nx, ny, nz, res)], w * particleMass);

                    // Also splat to Beta coefficients (1/rho later) - just raw mass for now
                    // Note: Paper uses specialized beta = 1/rho. We accumulate mass here first.
                    // Phase Field (reuse weight accumulators as per Sec 3.3)
                    // atomicAdd(&grid.d_faceXData[BETA_COEFF_X][...], w * particleMass);
                }

                // Face X Position (staggered)
                float3 faceXCenter = make_float3(nx * cellSize, (ny + 0.5f) * cellSize, (nz + 0.5f) * cellSize);
                float distSqX = (localPos.x - faceXCenter.x)*(localPos.x - faceXCenter.x) +
                                (localPos.y - faceXCenter.y)*(localPos.y - faceXCenter.y) +
                                (localPos.z - faceXCenter.z)*(localPos.z - faceXCenter.z);
                float wX = computeKernelWeight(distSqX, supportRadiusSq);
                if (wX > 0.0f) {
                    atomicAdd(&grid.d_faceXData[MOMENTUM_X][offsetFx + faceXIndex3D(nx, ny, nz, res)], wX * particleMass * vel.x);
                    atomicAdd(&grid.d_faceXData[MASS_DENSITY_X][offsetFx + faceXIndex3D(nx, ny, nz, res)], wX * particleMass);
                }

                // Face Y Position (staggered)
                float3 faceYCenter = make_float3((nx + 0.5f) * cellSize, ny * cellSize, (nz + 0.5f) * cellSize);
                float distSqY = (localPos.x - faceYCenter.x)*(localPos.x - faceYCenter.x) +
                                (localPos.y - faceYCenter.y)*(localPos.y - faceYCenter.y) +
                                (localPos.z - faceYCenter.z)*(localPos.z - faceYCenter.z);
                float wY = computeKernelWeight(distSqY, supportRadiusSq);
                if (wY > 0.0f) {
                    atomicAdd(&grid.d_faceYData[MOMENTUM_Y][offsetFy + faceYIndex3D(nx, ny, nz, res)], wY * particleMass * vel.y);
                    atomicAdd(&grid.d_faceYData[MASS_DENSITY_Y][offsetFy + faceYIndex3D(nx, ny, nz, res)], wY * particleMass);
                }

                // Face Z Position (staggered)
                float3 faceZCenter = make_float3((nx + 0.5f) * cellSize, (ny + 0.5f) * cellSize, nz * cellSize);
                float distSqZ = (localPos.x - faceZCenter.x)*(localPos.x - faceZCenter.x) +
                                (localPos.y - faceZCenter.y)*(localPos.y - faceZCenter.y) +
                                (localPos.z - faceZCenter.z)*(localPos.z - faceZCenter.z);
                float wZ = computeKernelWeight(distSqZ, supportRadiusSq);
                if (wZ > 0.0f) {
                    atomicAdd(&grid.d_faceZData[MOMENTUM_Z][offsetFz + faceZIndex3D(nx, ny, nz, res)], wZ * particleMass * vel.z);
                    atomicAdd(&grid.d_faceZData[MASS_DENSITY_Z][offsetFz + faceZIndex3D(nx, ny, nz, res)], wZ * particleMass);
                }
            }
        }
    }
}

// --- Normalize Kernel ---
// u = P / m (Eq. 3 from Paper)
__global__ void normalizeVelocity(
    MSBG grid,
    DataChannel momChan, DataChannel massChan, DataChannel velChan,
    int axis
) {
    uint32_t bIdx = blockIdx.x;
    if (bIdx >= grid.numBlocks) return;

    uint32_t level = grid.d_refinementMap[bIdx];
    if (level >= MAX_LEVELS) return;

    int res = c_blockLayouts[level].res;
    int numFaces = (axis == AXIS_X) ? (res + 1) * res * res :
                   (axis == AXIS_Y) ? res * (res + 1) * res :
                                      res * res * (res + 1);

    uint32_t offset = (axis == AXIS_X) ? grid.d_faceXOffsets[bIdx] :
                      (axis == AXIS_Y) ? grid.d_faceYOffsets[bIdx] :
                                         grid.d_faceZOffsets[bIdx];

    float* momData = (axis == AXIS_X) ? grid.d_faceXData[momChan] : (axis == AXIS_Y) ? grid.d_faceYData[momChan] : grid.d_faceZData[momChan];
    float* massData = (axis == AXIS_X) ? grid.d_faceXData[massChan] : (axis == AXIS_Y) ? grid.d_faceYData[massChan] : grid.d_faceZData[massChan];
    float* velData = (axis == AXIS_X) ? grid.d_faceXData[velChan] : (axis == AXIS_Y) ? grid.d_faceYData[velChan] : grid.d_faceZData[velChan];

    for (int i = threadIdx.x; i < numFaces; i += blockDim.x) {
        float m = massData[offset + i];
        if (m > 1e-6f) {
            velData[offset + i] = momData[offset + i] / m;
        } else {
            velData[offset + i] = 0.0f;
        }
    }
}

class P2GTransfer {
public:
    void run(MSBGManager& mg, float3* d_pos, float3* d_vel, uint32_t numParticles) {
        // 1. Clear Grid Data (Momentum & Mass Accumulators)
        // ... (cudaMemset logic or helper)
        // For brevity assuming data is cleared before this call in main loop

        int threads = 256;
        int blocks = (numParticles + threads - 1) / threads;

        // 2. Rasterize Mass & Momentum
        p2g<<<blocks, threads>>>(
            mg.grid, d_pos, d_vel, numParticles, 1.0f, 0.0f // mass 1.0 for now, radius calc inside
        );
        cudaDeviceSynchronize();

        // 3. Normalize Velocity (u = P/m)
        int blockCount = mg.grid.numBlocks;
        normalizeVelocity<<<blockCount, 256>>>(mg.grid, MOMENTUM_X, MASS_DENSITY_X, VELOCITY_X, AXIS_X);
        normalizeVelocity<<<blockCount, 256>>>(mg.grid, MOMENTUM_Y, MASS_DENSITY_Y, VELOCITY_Y, AXIS_Y);
        normalizeVelocity<<<blockCount, 256>>>(mg.grid, MOMENTUM_Z, MASS_DENSITY_Z, VELOCITY_Z, AXIS_Z);
        cudaDeviceSynchronize();
    }
};

#endif //ANITOWAVE_ADAPTIVE_P2G_CUH