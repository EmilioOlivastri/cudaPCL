#include "cudaPCL/cudaTransform.h"


// Forward‐declare the device kernel
extern __global__ void transformPointCloudKernel(float* source,
                                                 const float*  transformationMatrix,
                                                 unsigned int n_points);


void launchTransformPointCloud(unsigned int n_points, float* source,
                               const float* transformationMatrix, cudaStream_t stream) 
{
    const int threadsPerBlock = 512;
    const int blocks = (n_points + threadsPerBlock - 1) / threadsPerBlock;

    transformPointCloudKernel<<< blocks, threadsPerBlock, 0, stream >>>(
        source,
        transformationMatrix,
        n_points);
    
    checkCudaErrors(cudaGetLastError());

    return;
}