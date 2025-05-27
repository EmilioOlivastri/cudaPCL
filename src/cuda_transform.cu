#include <cuda_runtime.h>


__global__ void transformPointCloudKernel(float* output, const float* source,
                                          const float*  transformationMatrix,
                                          unsigned int* n_points) 
{
    // Assuming source is a point cloud with 3D points (x, y, z)
    // and transformationMatrix is a 4x4 matrix for transformation.
    // Each thread will process one point.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_points) return;

    int vertex_idx = idx * 3;
    float x = source[vertex_idx];
    float y = source[vertex_idx + 1];
    float z = source[vertex_idx + 2];

    // Apply the transformation matrix
    output[vertex_idx]     = transformationMatrix[0] * x + 
                             transformationMatrix[4] * y + 
                             transformationMatrix[8] * z + 
                             transformationMatrix[12];

    output[vertex_idx + 1] = transformationMatrix[1] * x + 
                             transformationMatrix[5] * y + 
                             transformationMatrix[9] * z + 
                             transformationMatrix[13];

    output[vertex_idx + 2] = transformationMatrix[2]  * x + 
                             transformationMatrix[6]  * y + 
                             transformationMatrix[10] * z + 
                             transformationMatrix[14];

    return;
}