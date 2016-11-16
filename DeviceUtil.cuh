#ifndef DEVICE_UTIL_CUH__
#define DEVICE_UTIL_CUH__

#include "CudaMath.h"

__device__ uint GetGlobalThreadId()
{
    const uint IdInBlock = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint BlockId = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint GlobalThreadId = BlockId * blockDim.z * blockDim.y * blockDim.x + IdInBlock;

//    if (GlobalThreadId > 180)
//    {
//        printf("(%4i) tid: %3i %3i %3i (%3i) - bid: %3i %3i %3i (%3i) - dims (%3i %3i %3i) X (%3i %3i %3i)\n",
//               GlobalThreadId, threadIdx.x, threadIdx.y, threadIdx.z, IdInBlock,
//               blockIdx.x, blockIdx.y, blockIdx.z, BlockId,
//               blockDim.x, blockDim.y, blockDim.z,
//               gridDim.x, gridDim.y, gridDim.z);
//    }

    return GlobalThreadId;
}

/**
 * @brief ClampLength Clamp vector to a maximal length
 * @param Vector
 * @param Max
 * @return The clamped vector
 */
inline __host__ __device__ float3 ClampLength(const float3& Vector, const float Max)
{
    const float InitialLength = Length(Vector);
    float3 NewVector;
    if (InitialLength > Max)
    {
        NewVector = Vector * (Max / InitialLength);
    }
    else
    {
        NewVector = Vector;
    }

    return NewVector;
}


#endif // DEVICE_UTIL_CUH__
