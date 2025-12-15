#ifndef ANITOWAVE_ADAPTIVE_VTX_VISUALIZATION_CUH
#define ANITOWAVE_ADAPTIVE_VTX_VISUALIZATION_CUH

#include <cstdio>
#include <vector>
#include <MSBG.cuh>

__global__ void sampleMSBG(MSBG grid, float* d_output, int3 outDims, float3 domainSize, DataChannel channel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outDims.x * outDims.y * outDims.z;
    if (idx >= total) return;

    int z = idx / (outDims.x * outDims.y);
    int rem = idx % (outDims.x * outDims.y);
    int y = rem / outDims.x;
    int x = rem % outDims.x;

    float3 pos;
    pos.x = (x + 0.5f) * (domainSize.x / outDims.x);
    pos.y = (y + 0.5f) * (domainSize.y / outDims.y);
    pos.z = (z + 0.5f) * (domainSize.z / outDims.z);

    int3 bCoord = worldToBlockCoord(pos, grid.blockSize, grid.domainMin);

    // Check bounds
    if (bCoord.x < 0 || bCoord.x >= grid.indexDims.x ||
        bCoord.y < 0 || bCoord.y >= grid.indexDims.y ||
        bCoord.z < 0 || bCoord.z >= grid.indexDims.z) {
        d_output[idx] = 0.0f;
        return;
        }

    uint32_t blockIdx = blockCoordToIndex(bCoord, grid.indexDims);
    uint32_t level = grid.d_refinementMap[blockIdx];

    // If block is inactive (or air), return 0
    if (level >= MAX_LEVELS) {
        d_output[idx] = 0.0f;
        return;
    }

    // Transform World Pos -> Block-Local Pos
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

    // Clamp to valid range (handles floating point epsilon issues at edges)
    cx = max(0, min(cx, res - 1));
    cy = max(0, min(cy, res - 1));
    cz = max(0, min(cz, res - 1));

    uint32_t offset = grid.d_cellOffsets[blockIdx];
    d_output[idx] = grid.d_cellData[channel][offset + cellIndex3D(cx, cy, cz, res)];
}

void exportUniformPressure_VTK(
    MSBGManager& mg,
    int3 res,               // Resolution of the output VTK (e.g., 256, 256, 256)
    const char* filename
) {
    size_t totalVoxels = res.x * res.y * res.z;
    float* d_uniform;
    cudaMalloc(&d_uniform, totalVoxels * sizeof(float));

    // Calculate domain size
    float3 domainSize = make_float3(
        mg.grid.indexDims.x * mg.grid.blockSize,
        mg.grid.indexDims.y * mg.grid.blockSize,
        mg.grid.indexDims.z * mg.grid.blockSize
    );

    int threads = 256;
    int blocks = (totalVoxels + threads - 1) / threads;

    // Run Sampling Kernel
    sampleMSBG<<<blocks, threads>>>(
        mg.grid,
        d_uniform,
        res,
        domainSize,
        PRESSURE
    );
    cudaDeviceSynchronize();

    // Copy to Host
    std::vector<float> h_data(totalVoxels);
    cudaMemcpy(h_data.data(), d_uniform, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_uniform);

    // Write VTK (Structured Points)
    FILE* fp = fopen(filename, "w");
    if(!fp) { printf("Error opening file %s\n", filename); return; }

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Uniform Pressure Resample\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", res.x, res.y, res.z);
    fprintf(fp, "ORIGIN 0 0 0\n");
    fprintf(fp, "SPACING %f %f %f\n",
        domainSize.x / res.x,
        domainSize.y / res.y,
        domainSize.z / res.z
    );

    fprintf(fp, "POINT_DATA %zu\n", totalVoxels);
    fprintf(fp, "SCALARS pressure float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for(size_t i=0; i<totalVoxels; i++) {
        fprintf(fp, "%f\n", h_data[i]);
    }

    fclose(fp);
    printf("Exported pressure field to %s (%dx%dx%d)\n", filename, res.x, res.y, res.z);
}

void exportMultigrid_VTK(const uint32_t* refinementMap, int3 dims, float blockSize, const char* filename) {
    FILE* fp = fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "MSBG Grid\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    uint32_t nx = dims.x;
    uint32_t ny = dims.y;
    uint32_t nz = dims.z;

    uint32_t numBlocks = nx * ny * nz;

    uint32_t numPoints = numBlocks * 8;
    uint32_t numCells = numBlocks;

    fprintf(fp, "POINTS %u float\n", numPoints);

    for (uint32_t z = 0; z < nz; z++) {
        for (uint32_t y = 0; y < ny; y++) {
            for (uint32_t x = 0; x < nx; x++) {
                float3 p = make_float3(x * blockSize, y * blockSize, z * blockSize);

                float3 corners[8] = {
                    make_float3(p.x, p.y, p.z),
                    make_float3(p.x + blockSize, p.y, p.z),
                    make_float3(p.x + blockSize, p.y + blockSize, p.z),
                    make_float3(p.x, p.y + blockSize, p.z),
                    make_float3(p.x, p.y, p.z + blockSize),
                    make_float3(p.x + blockSize, p.y, p.z + blockSize),
                    make_float3(p.x + blockSize, p.y + blockSize, p.z + blockSize),
                    make_float3(p.x, p.y + blockSize, p.z + blockSize)
                };

                for (int i = 0; i < 8; i++) {
                    fprintf(fp, "%f %f %f\n", corners[i].x, corners[i].y, corners[i].z);
                }
            }
        }
    }

    fprintf(fp, "CELLS %u %u\n", numCells, numCells * 9);

    for (uint32_t i = 0; i < numCells; i++) {
        uint32_t base = i * 8;
        fprintf(fp, "8 %u %u %u %u %u %u %u %u\n",
                base, base+1, base+2, base+3, base+4, base+5, base+6, base+7);
    }

    fprintf(fp, "CELL_TYPES %u\n", numCells);
    for (uint32_t i = 0; i < numCells; i++) {
        fprintf(fp, "12\n");
    }

    fprintf(fp, "CELL_DATA %u\n", numCells);
    fprintf(fp, "SCALARS refineLevel int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (uint32_t i = 0; i < numCells; i++) {
        uint32_t lv = refinementMap[i];
        if (lv == UINT32_MAX) lv = -1;
        fprintf(fp, "%d\n", lv);
    }

    fclose(fp);
}

void exportParticles_VTK(const std::vector<float3>& pos, const std::vector<float>& rad, const char* filename) {
    FILE* fp = fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Particle Cloud\n");
    fprintf(fp, "ASCII\n") ;
    fprintf(fp, "DATASET POLYDATA\n");
    fprintf(fp, "POINTS %zu float\n", pos.size());

    for (auto& p : pos) {
        fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
    }

    fprintf(fp, "POINT_DATA %zu\n", pos.size());
    fprintf(fp, "SCALARS radius float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto r : rad) {
        fprintf(fp, "%f\n", r);
    }

    fclose(fp);
}

#endif //ANITOWAVE_ADAPTIVE_VTX_VISUALIZATION_CUH