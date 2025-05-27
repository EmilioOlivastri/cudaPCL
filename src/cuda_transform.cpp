#include "cudaPCL/cudaTransform.h"


// Forward‐declare the device kernel
extern __global__ void transformPointCloudKernel(float* output, const float* source,
                                                 const float*  transformationMatrix,
                                                 unsigned int* n_points);


void launchTransformPointCloud(float* output, unsigned int* n_points, const float* source,
                               const float* transformationMatrix, cudaStream_t stream) 
{
    const int threadsPerBlock = 256;
    const int blocks = (*n_points + threadsPerBlock - 1) / threadsPerBlock;

    transformPointCloudKernel<<< blocks, threadsPerBlock, 0, stream >>>(
        output, source,
        transformationMatrix,
        n_points);
    
    checkCudaErrors(cudaGetLastError());

    return;
}