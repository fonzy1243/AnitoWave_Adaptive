#ifndef ANITOWAVE_ADAPTIVE_VTX_VISUALIZATION_CUH
#define ANITOWAVE_ADAPTIVE_VTX_VISUALIZATION_CUH

#include <cstdio>
#include <vector>

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