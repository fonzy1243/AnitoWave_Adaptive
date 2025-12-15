#ifndef ANITOWAVE_ADAPTIVE_G2P_CUH
#define ANITOWAVE_ADAPTIVE_G2P_CUH

#include <MSBG.cuh>
#include <VCycle.cuh>

// --- INTERPOLATION HELPERS ---

// Samples a specific velocity component (u, v, or w) at a world position
// using Trilinear Interpolation on the Staggered (MAC) grid.
__device__ inline float interpolateComponent(
    const MSBG& grid,
    float* data,
    uint32_t* offsets,
    float3 pos,
    int axis // 0=X, 1=Y, 2=Z
) {
    // 1. Find Block
    int3 bCoord = worldToBlockCoord(pos, grid.blockSize, grid.domainMin);
    uint32_t blockIdx = blockCoordToIndex(bCoord, grid.indexDims);

    // Boundary/Air check
    if (blockIdx >= grid.numBlocks || grid.d_refinementMap[blockIdx] >= MAX_LEVELS) return 0.0f;

    uint32_t level = grid.d_refinementMap[blockIdx];
    int res = c_blockLayouts[level].res;
    float cellSize = grid.blockSize / (float)res;

    // 2. Local Coordinate in Block
    float3 blockOrigin;
    blockOrigin.x = bCoord.x * grid.blockSize;
    blockOrigin.y = bCoord.y * grid.blockSize;
    blockOrigin.z = bCoord.z * grid.blockSize;

    // Staggered Offset: Velocity samples are shifted by half a cell on their respective axis
    float3 localPos = make_float3(pos.x - blockOrigin.x, pos.y - blockOrigin.y, pos.z - blockOrigin.z);
    if (axis == 0) localPos.y -= 0.5f * cellSize; localPos.z -= 0.5f * cellSize;
    if (axis == 1) localPos.x -= 0.5f * cellSize; localPos.z -= 0.5f * cellSize;
    if (axis == 2) localPos.x -= 0.5f * cellSize; localPos.y -= 0.5f * cellSize;

    // 3. Grid Index (Float)
    float fx = localPos.x / cellSize;
    float fy = localPos.y / cellSize;
    float fz = localPos.z / cellSize;

    // Floor to find bottom-left corner
    int ix = floorf(fx);
    int iy = floorf(fy);
    int iz = floorf(fz);

    // Fraction for interpolation
    float tx = fx - ix;
    float ty = fy - iy;
    float tz = fz - iz;

    // 4. Trilinear Sample
    // We use getVectorLevel (borrowed/duplicated logic) to handle crossing block boundaries safely
    float c000 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix,   iy,   iz  ), res, 0, 0, 0);
    float c100 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix+1, iy,   iz  ), res, 0, 0, 0);
    float c010 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix,   iy+1, iz  ), res, 0, 0, 0);
    float c110 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix+1, iy+1, iz  ), res, 0, 0, 0);
    float c001 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix,   iy,   iz+1), res, 0, 0, 0);
    float c101 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix+1, iy,   iz+1), res, 0, 0, 0);
    float c011 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix,   iy+1, iz+1), res, 0, 0, 0);
    float c111 = getVectorLevel(grid, data, offsets, blockIdx, make_int3(ix+1, iy+1, iz+1), res, 0, 0, 0);

    return lerp(
        lerp(lerp(c000, c100, tx), lerp(c010, c110, tx), ty),
        lerp(lerp(c001, c101, tx), lerp(c011, c111, tx), ty),
        tz
    );
}

// Wrapper to sample full velocity vector
__device__ inline float3 sampleVelocity(const MSBG& grid, float3 pos) {
    float u = interpolateComponent(grid, grid.d_faceXData[VELOCITY_X], grid.d_faceXOffsets, pos, 0);
    float v = interpolateComponent(grid, grid.d_faceYData[VELOCITY_Y], grid.d_faceYOffsets, pos, 1);
    float w = interpolateComponent(grid, grid.d_faceZData[VELOCITY_Z], grid.d_faceZOffsets, pos, 2);
    return make_float3(u, v, w);
}

// --- KERNELS ---

// FLIP Update: v_p = (1-alpha)*v_PIC + alpha*(v_old + (v_new_grid - v_old_grid))
// Simplified as: v_p += (v_grid_new - v_grid_old)  [Pure FLIP, alpha=1.0]
// Or mixed:      v_p = alpha * (v_p + du) + (1-alpha) * v_grid_new
__global__ void g2pUpdateKernel(
    MSBG grid,
    float3* p_vel,
    const float3* p_pos,
    uint32_t numParticles,
    float** d_velOldGrid, // Pointers to backup channels [X, Y, Z]
    float alphaFlip
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = p_pos[idx];
    float3 vp_old = p_vel[idx];

    // 1. Sample Current Grid Velocity (Divergence Free)
    float3 v_grid_new = sampleVelocity(grid, pos);

    // 2. Sample Old Grid Velocity (From Backup Channels)
    // Note: We need to temporarily swap the pointers in the MSBG struct or pass them explicitly.
    // Here we pass explicit pointers to the interpolate function.
    float u_old = interpolateComponent(grid, d_velOldGrid[0], grid.d_faceXOffsets, pos, 0);
    float v_old = interpolateComponent(grid, d_velOldGrid[1], grid.d_faceYOffsets, pos, 1);
    float w_old = interpolateComponent(grid, d_velOldGrid[2], grid.d_faceZOffsets, pos, 2);
    float3 v_grid_old = make_float3(u_old, v_old, w_old);

    // 3. Compute Change
    float3 dv = make_float3(
        v_grid_new.x - v_grid_old.x,
        v_grid_new.y - v_grid_old.y,
        v_grid_new.z - v_grid_old.z
    );

    // 4. FLIP Blend (Eq. 12)
    float3 pic_part = v_grid_new;
    float3 flip_part = make_float3(vp_old.x + dv.x, vp_old.y + dv.y, vp_old.z + dv.z);

    p_vel[idx] = make_float3(
        alphaFlip * flip_part.x + (1.0f - alphaFlip) * pic_part.x,
        alphaFlip * flip_part.y + (1.0f - alphaFlip) * pic_part.y,
        alphaFlip * flip_part.z + (1.0f - alphaFlip) * pic_part.z
    );
}

// RK3 Advection
__global__ void advectParticlesKernel(
    MSBG grid,
    float3* p_pos,
    uint32_t numParticles,
    float dt,
    float3 domainSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 x = p_pos[idx];

    // Step 1
    float3 v1 = sampleVelocity(grid, x);

    // Step 2
    float3 x2 = make_float3(x.x + 0.5f*dt*v1.x, x.y + 0.5f*dt*v1.y, x.z + 0.5f*dt*v1.z);
    float3 v2 = sampleVelocity(grid, x2);

    // Step 3
    float3 x3 = make_float3(x.x + 0.75f*dt*v2.x, x.y + 0.75f*dt*v2.y, x.z + 0.75f*dt*v2.z);
    float3 v3 = sampleVelocity(grid, x3);

    // Final Combine
    float3 v_final = make_float3(
        (2.f/9.f)*v1.x + (3.f/9.f)*v2.x + (4.f/9.f)*v3.x,
        (2.f/9.f)*v1.y + (3.f/9.f)*v2.y + (4.f/9.f)*v3.y,
        (2.f/9.f)*v1.z + (3.f/9.f)*v2.z + (4.f/9.f)*v3.z
    );

    float3 newPos = make_float3(
        x.x + dt * v_final.x,
        x.y + dt * v_final.y,
        x.z + dt * v_final.z
    );

    // Simple Domain Clamp
    newPos.x = fmaxf(0.1f, fminf(domainSize.x - 0.1f, newPos.x));
    newPos.y = fmaxf(0.1f, fminf(domainSize.y - 0.1f, newPos.y));
    newPos.z = fmaxf(0.1f, fminf(domainSize.z - 0.1f, newPos.z));

    p_pos[idx] = newPos;
}

class G2PManager {
public:
    float* d_backupVel[3]; // Stores u*, v*, w* before pressure solve

    void init(MSBGManager& mg) {
        // Allocate backup buffers same size as velocity channels
        cudaMalloc(&d_backupVel[0], mg.grid.totalFacesX * sizeof(float));
        cudaMalloc(&d_backupVel[1], mg.grid.totalFacesY * sizeof(float));
        cudaMalloc(&d_backupVel[2], mg.grid.totalFacesZ * sizeof(float));
    }

    void cleanup() {
        cudaFree(d_backupVel[0]);
        cudaFree(d_backupVel[1]);
        cudaFree(d_backupVel[2]);
    }

    void backupGridVelocity(MSBGManager& mg) {
        cudaMemcpy(d_backupVel[0], mg.grid.d_faceXData[VELOCITY_X], mg.grid.totalFacesX * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_backupVel[1], mg.grid.d_faceYData[VELOCITY_Y], mg.grid.totalFacesY * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_backupVel[2], mg.grid.d_faceZData[VELOCITY_Z], mg.grid.totalFacesZ * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    void performG2P(MSBGManager& mg, float3* p_vel, const float3* p_pos, uint32_t n, float alpha=0.97f) {
        float* d_ptrs[3];
        cudaMalloc(&d_ptrs[0], 3*sizeof(float*)); // GPU array of pointers
        cudaMemcpy(d_ptrs, d_backupVel, 3*sizeof(float*), cudaMemcpyHostToDevice);

        float* h_ptrs[3] = { d_backupVel[0], d_backupVel[1], d_backupVel[2] };
        float** d_dev_ptrs;
        cudaMalloc(&d_dev_ptrs, 3 * sizeof(float*));
        cudaMemcpy(d_dev_ptrs, h_ptrs, 3 * sizeof(float*), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        g2pUpdateKernel<<<blocks, threads>>>(mg.grid, p_vel, p_pos, n, d_dev_ptrs, alpha);
        cudaDeviceSynchronize();
        cudaFree(d_dev_ptrs);
    }

    void advect(MSBGManager& mg, float3* p_pos, uint32_t n, float dt) {
        float3 domain = make_float3(
            mg.grid.indexDims.x * mg.grid.blockSize,
            mg.grid.indexDims.y * mg.grid.blockSize,
            mg.grid.indexDims.z * mg.grid.blockSize
        );

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        advectParticlesKernel<<<blocks, threads>>>(mg.grid, p_pos, n, dt, domain);
        cudaDeviceSynchronize();
    }
};

#endif //ANITOWAVE_ADAPTIVE_G2P_CUH