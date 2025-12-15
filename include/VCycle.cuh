#ifndef ANITOWAVE_ADAPTIVE_VCYCLE_CUH
#define ANITOWAVE_ADAPTIVE_VCYCLE_CUH

#include <vector>
#include <MSBG.cuh>
#include <MultigridTransfer.cuh>
#include <Smoother.cuh>

// --- HELPER: Neighbor Access for Residual (Same as Smoother) ---
__device__ inline float getVectorLevel(
    const MSBG& grid,
    const float* __restrict__ vec_data,
    const uint32_t* __restrict__ offsets,
    uint32_t blockIdx,
    int3 localCoord,
    int currentRes,
    int dx, int dy, int dz
) {
    int3 blockCoord = grid.d_blockInfo[blockIdx].coord;
    int nx = localCoord.x + dx;
    int ny = localCoord.y + dy;
    int nz = localCoord.z + dz;

    // 1. Fast Path (Inside Block)
    if (nx >= 0 && nx < currentRes && ny >= 0 && ny < currentRes && nz >= 0 && nz < currentRes) {
        uint32_t offset = offsets[blockIdx];
        return vec_data[offset + cellIndex3D(nx, ny, nz, currentRes)];
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

    // Boundary Condition: Dirichlet (0)
    if (nBlockIdx == -1 || nBlockIdx >= grid.numBlocks) return 0.f;

    // 3. Ghost Value Retrieval
    uint32_t currentLevel = grid.d_refinementMap[blockIdx];
    uint32_t neighborLevel = grid.d_refinementMap[nBlockIdx];
    uint32_t nOffset = offsets[nBlockIdx];

    // Case A: Same Level
    if (currentLevel == neighborLevel) {
        return vec_data[nOffset + cellIndex3D(nx, ny, nz, currentRes)];
    }

    // Case B: Neighbor is Coarser (We are Fine)
    if (neighborLevel < currentLevel) {
        int scale = 1 << (currentLevel - neighborLevel);
        // Map ghost coordinate into coarse neighbor
        int ghostX = (dx == -1) ? (currentRes/scale - 1) : (dx == 1) ? 0 : (nx / scale);
        int ghostY = (dy == -1) ? (currentRes/scale - 1) : (dy == 1) ? 0 : (ny / scale);
        int ghostZ = (dz == -1) ? (currentRes/scale - 1) : (dz == 1) ? 0 : (nz / scale);
        int nRes = currentRes / scale;
        return vec_data[nOffset + cellIndex3D(ghostX, ghostY, ghostZ, nRes)];
    }

    // Case C: Neighbor is Finer (We are Coarse)
    if (neighborLevel > currentLevel) {
        int scale = 1 << (neighborLevel - currentLevel);
        int nRes = currentRes * scale;
        int ghostX = (dx == -1) ? (nRes - 1) : (dx == 1) ? 0 : (nx * scale + scale/2);
        int ghostY = (dy == -1) ? (nRes - 1) : (dy == 1) ? 0 : (ny * scale + scale/2);
        int ghostZ = (dz == -1) ? (nRes - 1) : (dz == 1) ? 0 : (nz * scale + scale/2);
        return vec_data[nOffset + cellIndex3D(ghostX, ghostY, ghostZ, nRes)];
    }

    return 0.f;
}

// --- RESIDUAL KERNEL (UPDATED) ---
__global__ void computeExplicitResidual(
    MSBG grid, // Passed by value for convenience (contains pointers)
    const uint32_t* __restrict__ cellOffsets,
    const uint32_t* __restrict__ fXOffsets, const float* __restrict__ betaX,
    const uint32_t* __restrict__ fYOffsets, const float* __restrict__ betaY,
    const uint32_t* __restrict__ fZOffsets, const float* __restrict__ betaZ,
    const float* __restrict__ x_vec,
    const float* __restrict__ b_vec,
    float* __restrict__ r_vec,
    int mgLevel
) {
    uint32_t bIdx = blockIdx.x;
    if (bIdx >= grid.numBlocks) return;

    uint32_t msbgLevel = grid.d_refinementMap[bIdx];
    int res = c_blockLayouts[msbgLevel].res >> mgLevel;
    if (res < 1) return;

    uint32_t offset = cellOffsets[bIdx];
    uint32_t fxOff = fXOffsets[bIdx];
    uint32_t fyOff = fYOffsets[bIdx];
    uint32_t fzOff = fZOffsets[bIdx];

    for (int i = threadIdx.x; i < res * res * res; i += blockDim.x) {
        int z = i / (res * res);
        int rem = i % (res * res);
        int y = rem / res;
        int x = rem % res;

        float bW = betaX[fxOff + faceXIndex3D(x, y, z, res)];
        float bE = betaX[fxOff + faceXIndex3D(x+1, y, z, res)];
        float bS = betaY[fyOff + faceYIndex3D(x, y, z, res)];
        float bN = betaY[fyOff + faceYIndex3D(x, y+1, z, res)];
        float bB = betaZ[fzOff + faceZIndex3D(x, y, z, res)];
        float bT = betaZ[fzOff + faceZIndex3D(x, y, z+1, res)];
        float diag = -(bW + bE + bS + bN + bB + bT);

        float valC = x_vec[offset + i];

        // CORRECTED: Use getVectorLevel to fetch neighbors across blocks
        float valW = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, -1, 0, 0);
        float valE = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, 1, 0, 0);
        float valS = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, 0, -1, 0);
        float valN = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, 0, 1, 0);
        float valB = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, 0, 0, -1);
        float valT = getVectorLevel(grid, x_vec, cellOffsets, bIdx, make_int3(x, y, z), res, 0, 0, 1);

        float Ax = diag * valC + bW*valW + bE*valE + bS*valS + bN*valN + bB*valB + bT*valT;
        r_vec[offset + i] = b_vec[offset + i] - Ax;
    }
}

__global__ void zeroVector(float* data, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 0.0f;
}

// --- V-CYCLE SOLVER CLASS ---

class VCycleSolver {
public:
    int maxLevels;
    AdaptiveSmoother smoother;
    MultigridTransfer transfer;

    // Hierarchy Arrays
    std::vector<float*> levels_x;
    std::vector<float*> levels_b;
    std::vector<float*> levels_beta_x;
    std::vector<float*> levels_beta_y;
    std::vector<float*> levels_beta_z;

    // Offset Arrays
    std::vector<uint32_t*> levels_c_off;
    std::vector<uint32_t*> levels_fx_off;
    std::vector<uint32_t*> levels_fy_off;
    std::vector<uint32_t*> levels_fz_off;

    std::vector<uint32_t> levels_total_cells;

    VCycleSolver(int levels, uint32_t numBlocks) : maxLevels(levels), smoother(numBlocks) {}

    ~VCycleSolver() {
        for (size_t i = 1; i < levels_x.size(); i++) {
            if (levels_x[i]) cudaFree(levels_x[i]);
            if (levels_b[i]) cudaFree(levels_b[i]);
            if (levels_beta_x[i]) cudaFree(levels_beta_x[i]);
            if (levels_beta_y[i]) cudaFree(levels_beta_y[i]);
            if (levels_beta_z[i]) cudaFree(levels_beta_z[i]);
            if (levels_c_off[i]) cudaFree(levels_c_off[i]);
            if (levels_fx_off[i]) cudaFree(levels_fx_off[i]);
            if (levels_fy_off[i]) cudaFree(levels_fy_off[i]);
            if (levels_fz_off[i]) cudaFree(levels_fz_off[i]);
        }
    }

    void init(MSBGManager& mg) {
        float* h_cellData[NUM_CHANNELS];
        float* h_faceXData[NUM_CHANNELS];
        float* h_faceYData[NUM_CHANNELS];
        float* h_faceZData[NUM_CHANNELS];

        cudaMemcpy(h_cellData, mg.grid.d_cellData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_faceXData, mg.grid.d_faceXData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_faceYData, mg.grid.d_faceYData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_faceZData, mg.grid.d_faceZData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);

        levels_x.push_back(h_cellData[PRESSURE]);
        levels_b.push_back(h_cellData[DIVERGENCE]);
        levels_beta_x.push_back(h_faceXData[BETA_COEFF_X]);
        levels_beta_y.push_back(h_faceYData[BETA_COEFF_Y]);
        levels_beta_z.push_back(h_faceZData[BETA_COEFF_Z]);

        levels_c_off.push_back(mg.grid.d_cellOffsets);
        levels_fx_off.push_back(mg.grid.d_faceXOffsets);
        levels_fy_off.push_back(mg.grid.d_faceYOffsets);
        levels_fz_off.push_back(mg.grid.d_faceZOffsets);
        levels_total_cells.push_back(mg.grid.totalCells);

        for (int l = 1; l < maxLevels; l++) {
            uint32_t totalC = 0, totalFx = 0, totalFy = 0, totalFz = 0;
            uint32_t *d_c, *d_fx, *d_fy, *d_fz;

            cudaMalloc(&d_c, mg.grid.numBlocks * sizeof(uint32_t));
            cudaMalloc(&d_fx, mg.grid.numBlocks * sizeof(uint32_t));
            cudaMalloc(&d_fy, mg.grid.numBlocks * sizeof(uint32_t));
            cudaMalloc(&d_fz, mg.grid.numBlocks * sizeof(uint32_t));

            computeCoarseOffsets(mg, l, d_c, totalC);
            levels_c_off.push_back(d_c);
            levels_total_cells.push_back(totalC);

            computeCoarseFaceOffsets(mg, l, AXIS_X, d_fx, totalFx);
            levels_fx_off.push_back(d_fx);

            computeCoarseFaceOffsets(mg, l, AXIS_Y, d_fy, totalFy);
            levels_fy_off.push_back(d_fy);

            computeCoarseFaceOffsets(mg, l, AXIS_Z, d_fz, totalFz);
            levels_fz_off.push_back(d_fz);

            float *dx, *db, *dbx, *dby, *dbz;
            cudaMalloc(&dx, totalC * sizeof(float));
            cudaMalloc(&db, totalC * sizeof(float));
            cudaMalloc(&dbx, totalFx * sizeof(float));
            cudaMalloc(&dby, totalFy * sizeof(float));
            cudaMalloc(&dbz, totalFz * sizeof(float));

            cudaMemset(dx, 0, totalC * sizeof(float));
            cudaMemset(db, 0, totalC * sizeof(float));
            cudaMemset(dbx, 0, totalFx * sizeof(float));
            cudaMemset(dby, 0, totalFy * sizeof(float));
            cudaMemset(dbz, 0, totalFz * sizeof(float));

            levels_x.push_back(dx);
            levels_b.push_back(db);
            levels_beta_x.push_back(dbx);
            levels_beta_y.push_back(dby);
            levels_beta_z.push_back(dbz);

            mg.runGalerkin(
                levels_beta_x[l-1], levels_beta_y[l-1], levels_beta_z[l-1],
                levels_fx_off[l-1], levels_fy_off[l-1], levels_fz_off[l-1],
                levels_beta_x[l], levels_beta_y[l], levels_beta_z[l],
                levels_fx_off[l], levels_fy_off[l], levels_fz_off[l],
                l
            );
        }
    }

    void runCycle(MSBGManager& mg, int level) {
        if (level == maxLevels - 1) {
            smoother.solveLevel(mg, level, levels_x[level], levels_b[level],
                levels_beta_x[level], levels_beta_y[level], levels_beta_z[level],
                levels_c_off[level], levels_fx_off[level], levels_fy_off[level], levels_fz_off[level], 50);
            return;
        }

        // 1. Pre-Smooth
        smoother.solveLevel(mg, level, levels_x[level], levels_b[level],
            levels_beta_x[level], levels_beta_y[level], levels_beta_z[level],
            levels_c_off[level], levels_fx_off[level], levels_fy_off[level], levels_fz_off[level], 2);

        // 2. Compute Residual
        float* d_r_temp;
        cudaMalloc(&d_r_temp, levels_total_cells[level] * sizeof(float));

        int threads = 256;
        int blocks = (mg.grid.numBlocks * 256 + threads - 1) / threads;

        // UPDATED LAUNCH: Passed MG Grid by value
        computeExplicitResidual<<<mg.grid.numBlocks, 256>>>(
            mg.grid, // Passed MSBG struct
            levels_c_off[level],
            levels_fx_off[level], levels_beta_x[level],
            levels_fy_off[level], levels_beta_y[level],
            levels_fz_off[level], levels_beta_z[level],
            levels_x[level], levels_b[level], d_r_temp, level
        );

        // 3. Restrict
        transfer.restrictResiduals(mg, d_r_temp, levels_b[level+1],
            levels_c_off[level], levels_c_off[level+1], level);

        cudaFree(d_r_temp);

        // 4. Zero Coarse Guess
        uint32_t coarseSize = levels_total_cells[level+1];
        if (coarseSize > 0) {
             zeroVector<<<(coarseSize+255)/256, 256>>>(levels_x[level+1], coarseSize);
        }

        // 5. Recurse
        runCycle(mg, level + 1);

        // 6. Prolongate
        transfer.prolongateAndCorrect(mg, levels_x[level], levels_x[level+1],
            levels_c_off[level], levels_c_off[level+1], level);

        // 7. Post-Smooth
        smoother.solveLevel(mg, level, levels_x[level], levels_b[level],
            levels_beta_x[level], levels_beta_y[level], levels_beta_z[level],
            levels_c_off[level], levels_fx_off[level], levels_fy_off[level], levels_fz_off[level], 2);
    }
};

#endif // ANITOWAVE_ADAPTIVE_VCYCLE_CUH