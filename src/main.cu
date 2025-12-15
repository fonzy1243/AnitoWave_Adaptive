#include <iostream>
#include <vector>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <MSBG.cuh>
#include <P2G.cuh>
#include <Smoother.cuh>
#include <VCycle.cuh>
#include <vtx_visualization.cuh>

struct SquareOp {
    __device__ float operator()(float x) const {
        return x * x;
    }
};

__global__ void test_rasterizeParticlesToRHS(
    MSBG grid,
    const float3* particles,
    uint32_t numParticles,
    DataChannel targetChannel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = particles[idx];

    // 1. Find Block
    int3 bCoord = worldToBlockCoord(pos, grid.blockSize, grid.domainMin);

    // Bounds check
    if (bCoord.x < 0 || bCoord.x >= grid.indexDims.x ||
        bCoord.y < 0 || bCoord.y >= grid.indexDims.y ||
        bCoord.z < 0 || bCoord.z >= grid.indexDims.z) return;

    uint32_t blockIdx = blockCoordToIndex(bCoord, grid.indexDims);
    uint32_t level = grid.d_refinementMap[blockIdx];

    // Ignore inactive blocks
    if (level >= MAX_LEVELS) return;

    // 2. Find Cell inside Block
    float3 blockOrigin;
    blockOrigin.x = bCoord.x * grid.blockSize;
    blockOrigin.y = bCoord.y * grid.blockSize;
    blockOrigin.z = bCoord.z * grid.blockSize;

    float3 localPos = make_float3(pos.x - blockOrigin.x, pos.y - blockOrigin.y, pos.z - blockOrigin.z);

    int res = c_blockLayouts[level].res;
    float cellSize = grid.blockSize / (float)res;

    int cx = (int)(localPos.x / cellSize);
    int cy = (int)(localPos.y / cellSize);
    int cz = (int)(localPos.z / cellSize);

    // Clamp
    cx = max(0, min(cx, res - 1));
    cy = max(0, min(cy, res - 1));
    cz = max(0, min(cz, res - 1));

    // 3. Splat Mass (Atomic Add)
    // We add a simplified mass of 1.0 per particle.
    // This creates a density field where particles are clustered.
    uint32_t offset = grid.d_cellOffsets[blockIdx];
    atomicAdd(&grid.d_cellData[targetChannel][offset + cellIndex3D(cx, cy, cz, res)], 1.0f);
}

__global__ void countActiveBlocks(const uint32_t* map, uint32_t* out, uint32_t N) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (map[i] <= (MAX_LEVELS - 1)) atomicAdd(out, 1);
}

__global__ void initArrayConst(float* data, float value, uint32_t N) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = value;
}

__global__ void initDivergence(float* div, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    div[idx] = sinf(static_cast<float>(idx) * 0.01f) * 0.1f;
}

int main() {
    std::cout << "CUDA MSBG Test" << std::endl;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA device " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Setup multigrid
    int3 dims = make_int3(60, 60, 60);
    float blockSize = 1.0f;
    float3 domainMin = make_float3(0, 0, 0);

    MSBGManager mg(dims, blockSize, domainMin);

    // Generate random particles
    const uint32_t numParticles = 50000;

    std::vector<float3> h_pos(numParticles);
    std::vector<float> h_rad(numParticles);

    uint32_t numBlocks = dims.x * dims.y * dims.z;

    std::vector<uint32_t> h_refinementMap(numBlocks);

    float3 center = make_float3(
        dims.x * blockSize * 0.5,
        dims.y * blockSize * 0.5,
        dims.z * blockSize * 0.5
    );

    float sphereRadius = dims.x * blockSize * 0.2f;

    for (uint32_t i = 0; i < numParticles; i++) {
        float x, y, z;

        while (true) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float rz = (float)rand() / RAND_MAX * 2.0f - 1.0f;

            float len2 = rx*rx + ry*ry + rz*rz;
            if (len2 <= 1.0f) {
                x = center.x + rx * sphereRadius;
                y = center.y + ry * sphereRadius;
                z = center.z + rz * sphereRadius;
                break;
            }
        }

        h_pos[i] = make_float3(x, y, z);
        h_rad[i] = 0.02f + 0.02f * ((float)rand() / RAND_MAX);
    }

    float3* d_pos;
    float3* d_vel;
    float* d_rad;
    cudaMalloc(&d_pos, numParticles * sizeof(float3));
    cudaMalloc(&d_rad, numParticles * sizeof(float));
    cudaMalloc(&d_vel, numParticles * sizeof(float3));
    cudaMemset(d_vel, 0, numParticles * sizeof(float3));
    cudaMemcpy(d_pos, h_pos.data(), numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rad, h_rad.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemset(mg.grid.d_refinementMap, MAX_LEVELS - 1, mg.grid.numBlocks * sizeof(uint8_t));

    int threads_map = 256;
    int blocks_map = (mg.grid.numBlocks + threads_map - 1) / threads_map;

    initRefinementMap<<<blocks_map, threads_map>>>(
        mg.grid.d_refinementMap,
        MAX_LEVELS - 1,
        mg.grid.numBlocks
    );
    cudaDeviceSynchronize();

    std::cout << "Marking active blocks..." << std::endl;

    int threads = 256;
    int blocks = (numParticles + threads - 1) / threads;

    markActiveBlocks<<<blocks, threads>>>(
        d_pos,
        d_rad,
        numParticles,
        mg.grid.d_refinementMap,
        mg.grid.indexDims,
        mg.grid.blockSize,
        mg.grid.domainMin
    );
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    std::cout << "Enforcing grading..." << std::endl;

    dim3 t(4, 4, 4);
    dim3 g(
        (dims.x + t.x - 1) / t.x,
        (dims.y + t.y - 1) / t.y,
        (dims.z + t.z - 1) / t.z
    );

    for (int iter = 0; iter < 3; iter++) {
        enforceGrading<<<g, t>>>(mg.grid.d_refinementMap, dims);
        cudaPeekAtLastError();
        cudaDeviceSynchronize();
    }

    uint32_t* d_activeCount;
    cudaMalloc(&d_activeCount, sizeof(uint32_t));
    cudaMemset(d_activeCount, 0, sizeof(uint32_t));

    int nb = (mg.grid.numBlocks + threads - 1) / threads;
    countActiveBlocks<<<nb, threads>>>(mg.grid.d_refinementMap, d_activeCount, mg.grid.numBlocks);
    cudaDeviceSynchronize();

    uint32_t h_activeCount = 0;
    cudaMemcpy(&h_activeCount, d_activeCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_activeCount);

    std::cout << "Active blocks: " << h_activeCount << " / " << mg.grid.numBlocks << std::endl;

    std::cout << "Computing cell/face counts..." << std::endl;

    uint32_t* d_cellCounts;
    uint32_t* d_fxCounts;
    uint32_t* d_fyCounts;
    uint32_t* d_fzCounts;

    cudaMalloc(&d_cellCounts, mg.grid.numBlocks * sizeof(uint32_t));
    cudaMalloc(&d_fxCounts, mg.grid.numBlocks * sizeof(uint32_t));
    cudaMalloc(&d_fyCounts, mg.grid.numBlocks * sizeof(uint32_t));
    cudaMalloc(&d_fzCounts, mg.grid.numBlocks * sizeof(uint32_t));

    countBlockElements<<<nb, threads>>>(
        mg.grid.d_refinementMap,
        d_cellCounts, d_fxCounts, d_fyCounts, d_fzCounts,
        mg.grid.numBlocks
    );
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_c(mg.grid.numBlocks);
    std::vector<uint32_t> h_fx(mg.grid.numBlocks);
    std::vector<uint32_t> h_fy(mg.grid.numBlocks);
    std::vector<uint32_t> h_fz(mg.grid.numBlocks);

    cudaMemcpy(h_c.data(), d_cellCounts, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx.data(), d_fxCounts, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fyCounts, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fzCounts, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_refinementMap.data(), mg.grid.d_refinementMap, mg.grid.numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    mg.grid.totalCells  = 0;
    mg.grid.totalFacesX = 0;
    mg.grid.totalFacesY = 0;
    mg.grid.totalFacesZ = 0;

    for (uint32_t i = 0; i < mg.grid.numBlocks; i++)
    {
        mg.grid.totalCells  += h_c[i];
        mg.grid.totalFacesX += h_fx[i];
        mg.grid.totalFacesY += h_fy[i];
        mg.grid.totalFacesZ += h_fz[i];
    }

    std::cout << "Total cell count: " << mg.grid.totalCells << "\n";
    std::cout << "Total faceX count: " << mg.grid.totalFacesX << "\n";
    std::cout << "Total faceY count: " << mg.grid.totalFacesY << "\n";
    std::cout << "Total faceZ count: " << mg.grid.totalFacesZ << "\n";

    thrust::device_ptr<uint32_t> dev_c_counts(d_cellCounts);
    thrust::device_ptr<uint32_t> dev_fx_counts(d_fxCounts);
    thrust::device_ptr<uint32_t> dev_fy_counts(d_fyCounts);
    thrust::device_ptr<uint32_t> dev_fz_counts(d_fzCounts);

    thrust::device_ptr<uint32_t> dev_c_offsets(mg.grid.d_cellOffsets);
    thrust::device_ptr<uint32_t> dev_fx_offsets(mg.grid.d_faceXOffsets);
    thrust::device_ptr<uint32_t> dev_fy_offsets(mg.grid.d_faceYOffsets);
    thrust::device_ptr<uint32_t> dev_fz_offsets(mg.grid.d_faceZOffsets);

    thrust::exclusive_scan(dev_c_counts, dev_c_counts + mg.grid.numBlocks, dev_c_offsets);
    thrust::exclusive_scan(dev_fx_counts, dev_fx_counts + mg.grid.numBlocks, dev_fx_offsets);
    thrust::exclusive_scan(dev_fy_counts, dev_fy_counts + mg.grid.numBlocks, dev_fy_offsets);
    thrust::exclusive_scan(dev_fz_counts, dev_fz_counts + mg.grid.numBlocks, dev_fz_offsets);

    cudaDeviceSynchronize();

    std::cout << "Allocating Data Channels...\n";
    mg.allocateChannels();

    float* h_faceXPtrs[NUM_CHANNELS];
    float* h_faceYPtrs[NUM_CHANNELS];
    float* h_faceZPtrs[NUM_CHANNELS];

    cudaMemcpy(h_faceXPtrs, mg.grid.d_faceXData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_faceYPtrs, mg.grid.d_faceYData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_faceZPtrs, mg.grid.d_faceZData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);

    float* d_fineBetaX = h_faceXPtrs[BETA_COEFF_X];
    float* d_fineBetaY = h_faceYPtrs[BETA_COEFF_Y];
    float* d_fineBetaZ = h_faceZPtrs[BETA_COEFF_Z];

    int numThreads = 256;
    int numBlocksInitX = (mg.grid.totalFacesX + numThreads - 1) / numThreads;
    initArrayConst<<<numBlocksInitX, numThreads>>>(h_faceXPtrs[BETA_COEFF_X], 1.f, mg.grid.totalFacesX);
    initArrayConst<<<(mg.grid.totalFacesY + 255)/256, 256>>>(h_faceYPtrs[BETA_COEFF_Y], 1.f, mg.grid.totalFacesY);
    initArrayConst<<<(mg.grid.totalFacesZ + 255)/256, 256>>>(h_faceZPtrs[BETA_COEFF_Z], 1.f, mg.grid.totalFacesZ);

    float* h_cellPtrs[NUM_CHANNELS];
    cudaMemcpy(h_cellPtrs, mg.grid.d_cellData, NUM_CHANNELS * sizeof(float*), cudaMemcpyDeviceToHost);
    initArrayConst<<<(mg.grid.totalCells + 255)/256, 256>>>(h_cellPtrs[PRESSURE], 0.f, mg.grid.totalCells);
    initArrayConst<<<(mg.grid.totalCells + 255)/256, 256>>>(h_cellPtrs[DIVERGENCE], 0.f, mg.grid.totalCells);
    cudaDeviceSynchronize();

    P2GTransfer p2g;
    p2g.run(mg, d_pos, d_vel, numParticles);

    // 2. Setup V-Cycle
    std::cout << "Building V-Cycle Hierarchy..." << std::endl;
    VCycleSolver solver(MAX_LEVELS, mg.grid.numBlocks);
    solver.init(mg);

    // 3. Run
    std::cout << "Running 10 V-Cycles..." << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << "Iteration " << i << std::endl;
        solver.runCycle(mg, 0);
    }
    cudaDeviceSynchronize();

    // 4. Verify
    thrust::device_ptr<float> dev_p(h_cellPtrs[PRESSURE]);
    float max_p = thrust::reduce(thrust::device, dev_p, dev_p + mg.grid.totalCells, -1e20f, thrust::maximum<float>());
    float min_p = thrust::reduce(thrust::device, dev_p, dev_p + mg.grid.totalCells, 1e20f, thrust::minimum<float>());
    std::cout << "Pressure Range: [" << min_p << ", " << max_p << "]" << std::endl;

    cudaFree(d_cellCounts);
    cudaFree(d_fxCounts);
    cudaFree(d_fyCounts);
    cudaFree(d_fzCounts);

    cudaFree(d_pos);
    cudaFree(d_rad);

    exportMultigrid_VTK(h_refinementMap.data(), dims, blockSize, "msbg.vtk");
    exportParticles_VTK(h_pos, h_rad, "particles.vtk");
    exportUniformPressure_VTK(mg, make_int3(512, 512, 512), "pressure_field.vtk");

    return 0;
}