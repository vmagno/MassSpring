#ifndef HOSTDEVICECODE_H
#define HOSTDEVICECODE_H

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Common.h"

#define CudaCheck(call) { CudaCallWithCheck((call), __FILE__, __LINE__, #call); }

inline void CudaCallWithCheck(cudaError_t ReturnCode, const char* Filename, int LineNumber, const char* LineCode, bool bDoAbort = true)
{
    if (ReturnCode != cudaSuccess)
    {
        fprintf(stderr, "[ERROR] %s (%s: %d)\n        \"%s\"\n", cudaGetErrorString(ReturnCode), Filename, LineNumber, LineCode);
        if (bDoAbort) exit(ReturnCode);
    }
}

struct MSSParameters
{
    float Gravity;
    float FloorHeight;
    float FloorStrength;
    float Damping;
    float YoungModulus;
    float PoissonRatio;
    float TotalMass;
    bool bFixedTop;

    float DeltaT;
    uint NumParticles;
    uint NumSprings;

    void init()
    {
        Gravity = 9.81f;
        FloorHeight = 0.f;
        FloorStrength = 20000.f;
        Damping = 0.98f;
        YoungModulus = 2000.f;
        PoissonRatio = 0.3f;
        TotalMass = 0.15f;
        bFixedTop = false;

        DeltaT = 0.f;
    }

};


struct Arrays
{
    // Particles
    float3* Positions;
    float3* Velocities;
    float3* Forces;
    float* ParticleMasses;
    float4* ParticleColors;

    // Springs
    uint2* SpringConnect;
    float* RestLengths;
    float* Coefficients;
    float4* SpringColors;

#ifdef DEBUG_VELOCITY
    float3* VelocityVectors;
    float4* VelocityColors;
#endif

    void ClearPtrs()
    {
        Positions = NULL;
        Velocities = NULL;
        Forces = NULL;
        ParticleMasses = NULL;
        ParticleColors = NULL;

        SpringConnect = NULL;
        RestLengths = NULL;
        Coefficients = NULL;
        SpringColors = NULL;
    }

    void FreeDeviceMem()
    {
        if (Positions) { CudaCheck(cudaFree(Positions)); }
        if (Velocities) { CudaCheck(cudaFree(Velocities)); }
        if (Forces) { CudaCheck(cudaFree(Forces)); }
        if (ParticleMasses) { CudaCheck(cudaFree(ParticleMasses)); }
        if (ParticleColors) { CudaCheck(cudaFree(ParticleColors)); }
        if (SpringConnect) { CudaCheck(cudaFree(SpringConnect)); }
        if (RestLengths) { CudaCheck(cudaFree(RestLengths)); }
        if (Coefficients) { CudaCheck(cudaFree(Coefficients)); }
        if (SpringColors) { CudaCheck(cudaFree(SpringColors)); }

        ClearPtrs();
    }
};

#endif // HOSTDEVICECODE_H
