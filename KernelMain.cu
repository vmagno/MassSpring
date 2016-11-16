#include "Kernels.h"

#include <cmath>
#include <cstdio>

#include "CudaMath.h"
#include "DeviceUtil.cuh"
#include "HostDeviceCode.h"

__global__ void TestKernel(uint* SomeValue)
{
    printf("SomePointer = %d\n", *SomeValue);
    SomeValue[0] = 5;
    printf("SomePointer = %d\n", *SomeValue);
}

void LaunchTest(uint SomeValue)
{
    printf("Launching kernel\n");
    uint* test_d;
    CudaCheck(cudaMalloc(&test_d, sizeof(uint)));
    cudaMemcpy(test_d, &SomeValue, sizeof(uint), cudaMemcpyHostToDevice);
    TestKernel <<<1, 1>>> (test_d);

    uint test;
    cudaMemcpy(&test, test_d, sizeof(uint), cudaMemcpyDeviceToHost);
    CudaCheck(cudaFree(test_d));

    fflush(stdout);
}

__global__ void ApplyForcesKernel(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ParticleId = GetGlobalThreadId();

    if (ParticleId >= Param.NumParticles) return;

    const float3 Position = DeviceArrays.Positions[ParticleId];

    // Gravity component
    float3 NewForce = make_float3(0.f, 0.f, -Param.Gravity);

    if (Position.z < Param.FloorHeight)
    {
        NewForce.z += (Param.FloorHeight - Position.z) * Param.FloorStrength;
    }

    DeviceArrays.Forces[ParticleId] = NewForce;
}

void LaunchApplyForces(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ThreadsPerBlock = 128;
    const dim3 BlockSize(ThreadsPerBlock);
    dim3 GridSize(uint(ceilf((float)Param.NumParticles / ThreadsPerBlock)));
    while (GridSize.x > 65535)
    {
        GridSize.x = uint(ceilf(GridSize.x / 2.f));
        GridSize.y *= 2;
    }

    ApplyForcesKernel<<<GridSize, BlockSize>>>(Param, DeviceArrays);
    CudaCheck(cudaThreadSynchronize());

//    printf("Apply forces done %i %i %i by %i %i %i, NumParticles = %i\n",
//           GridSize.x, GridSize.y, GridSize.z,
//           BlockSize.x, BlockSize.y, BlockSize.z,
//           Param.NumParticles);
    fflush(stdout);
}

__global__ void AddSpringForcesKernel(MSSParameters Param, Arrays DeviceArrays)
{
    const uint SpringId = GetGlobalThreadId();

    if (SpringId >= Param.NumSprings) return;

    const uint Pid1 = DeviceArrays.SpringConnect[SpringId].x;
    const uint Pid2 = DeviceArrays.SpringConnect[SpringId].y;

//    {
//        float3 pos1 = DeviceArrays.Positions[Pid1];
//        float3 pos2 = DeviceArrays.Positions[Pid2];
//        printf("part (%i, %i), pos(%5.2f, %5.2f, %5.2f - %5.2f, %5.2f, %5.2f) \n",
//               Pid1, Pid2, pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z);
//    }

    const float3 Vector = DeviceArrays.Positions[Pid1] - DeviceArrays.Positions[Pid2];
    const float Distance = Length(Vector);
    const float3 Direction = Vector / Distance;
    const float DeltaLength = DeviceArrays.RestLengths[SpringId] - Distance;
    const float3 BaseForce = Direction * (DeltaLength * DeviceArrays.Coefficients[SpringId]);

    const float p1Mass = DeviceArrays.ParticleMasses[Pid1];
    const float p2Mass = DeviceArrays.ParticleMasses[Pid2];

    atomicAdd(&DeviceArrays.Forces[Pid1].x, BaseForce.x / p1Mass);
    atomicAdd(&DeviceArrays.Forces[Pid1].y, BaseForce.y / p1Mass);
    atomicAdd(&DeviceArrays.Forces[Pid1].z, BaseForce.z / p1Mass);
    atomicAdd(&DeviceArrays.Forces[Pid2].x, -BaseForce.x / p2Mass);
    atomicAdd(&DeviceArrays.Forces[Pid2].y, -BaseForce.y / p2Mass);
    atomicAdd(&DeviceArrays.Forces[Pid2].z, -BaseForce.z / p2Mass);
}

void LaunchAddSpringForces(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ThreadsPerBlock = 128;
    const dim3 BlockSize(ThreadsPerBlock);
    dim3 GridSize(uint(ceilf((float)Param.NumSprings / ThreadsPerBlock)));
    while (GridSize.x > 65535)
    {
        GridSize.x = uint(ceilf(GridSize.x / 2.f));
        GridSize.y *= 2;
    }

    AddSpringForcesKernel<<<GridSize, BlockSize>>>(Param, DeviceArrays);

    CudaCheck(cudaThreadSynchronize());
//    printf("Add spring forces done %i %i %i by %i %i %i, NumSprings = %i\n",
//           GridSize.x, GridSize.y, GridSize.z,
//           BlockSize.x, BlockSize.y, BlockSize.z,
//           Param.NumSprings);
    fflush(stdout);

}

__global__ void IntegrateSystemKernel(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ParticleId = GetGlobalThreadId();

    if (ParticleId >= Param.NumParticles) return;

    float3 NewVelocity =
            ClampLength(DeviceArrays.Velocities[ParticleId] + (DeviceArrays.Forces[ParticleId] * Param.DeltaT),
                        5.f);

    DeviceArrays.Positions[ParticleId] =
            DeviceArrays.Positions[ParticleId] + NewVelocity * Param.DeltaT;

    NewVelocity = NewVelocity * Param.Damping;

    DeviceArrays.Velocities[ParticleId] = NewVelocity;
}

#ifdef DEBUG_VELOCITY
__global__ void UpdateDisplayedVelocities(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ParticleId = GetGlobalThreadId();

    if (ParticleId >= Param.NumParticles) return;

    DeviceArrays.VelocityVectors[ParticleId*2] = DeviceArrays.Positions[ParticleId];
    DeviceArrays.VelocityVectors[ParticleId*2 + 1] = DeviceArrays.Positions[ParticleId] + DeviceArrays.Velocities[ParticleId] * 0.2f;
    //DeviceArrays.VelocityVectors[ParticleId*2 + 1] = DeviceArrays.Positions[ParticleId] + make_float3(0.05f, 0.f, 0.f) * 1.f;

    DeviceArrays.VelocityColors[ParticleId*2] = make_float4(1.f);
    DeviceArrays.VelocityColors[ParticleId*2 + 1] = make_float4(1.f);
}
#endif // DEBUG_VELOCITY

void LaunchIntegrateSystem(MSSParameters Param, Arrays DeviceArrays)
{
    const uint ThreadsPerBlock = 128;
    const dim3 BlockSize(ThreadsPerBlock);
    dim3 GridSize(uint(ceilf((float)Param.NumParticles / ThreadsPerBlock)));
    while (GridSize.x > 65535)
    {
        GridSize.x = uint(ceilf(GridSize.x / 2.f));
        GridSize.y *= 2;
    }

    IntegrateSystemKernel<<<GridSize, BlockSize>>>(Param, DeviceArrays);

#ifdef DEBUG_VELOCITY
    UpdateDisplayedVelocities<<<GridSize, BlockSize>>>(Param, DeviceArrays);
#endif

    CudaCheck(cudaThreadSynchronize());

//    printf("Integrate done %i %i %i by %i %i %i, NumParticles = %i\n",
//           GridSize.x, GridSize.y, GridSize.z,
//           BlockSize.x, BlockSize.y, BlockSize.z,
//           Param.NumParticles);
    fflush(stdout);
}
