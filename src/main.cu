#include <iostream>
#include <vector>
#include <multigrid.cuh>
#include <vtx_visualization.cuh>

__global__ void countActiveBlocks(const uint32_t* map, uint32_t* out, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (map[i] <= (MAX_LEVELS - 1)) atomicAdd(out, 1);
}

int main() {
    std::cout << "CUDA MSBG Test" << std::endl;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA device " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Setup multigrid
    int3 dims = make_int3(80, 80, 80);
    float blockSize = 1.0f;
    float3 domainMin = make_float3(0, 0, 0);

    MultigridManager mg(dims, blockSize, domainMin);

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
    float* d_rad;
    cudaMalloc(&d_pos, numParticles * sizeof(float3));
    cudaMalloc(&d_rad, numParticles * sizeof(float));
    cudaMemcpy(d_pos, h_pos.data(), numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rad, h_rad.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(mg.grid.d_refinementMap, MAX_LEVELS - 1, mg.grid.numBlocks * sizeof(uint8_t));

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
    std::cout << std::endl;

    cudaFree(d_cellCounts);
    cudaFree(d_fxCounts);
    cudaFree(d_fyCounts);
    cudaFree(d_fzCounts);

    cudaFree(d_pos);
    cudaFree(d_rad);

    exportMultigrid_VTK(h_refinementMap.data(), dims, blockSize, "msbg.vtk");
    exportParticles_VTK(h_pos, h_rad, "particles.vtk");

    return 0;
}