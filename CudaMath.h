#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <cuda_runtime.h>

//////////////////////////
/// float2
/////////////////////////
inline __host__ __device__ float Dot(const float2& a, const float2& b)
{
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float Length(const float2& Vector)
{
    return sqrtf(Dot(Vector, Vector));
}

//////////////////////////
/// uint3
/////////////////////////
inline __host__ __device__ uint3 make_uint3(const uint a)
{
    return make_uint3(a, a, a);
}

inline __host__ __device__ uint3 operator+(const uint3& a, const uint3& b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/////////////////////////
/// float3
/////////////////////////

inline __host__ __device__ float3 abs(const float3& a)
{
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator/(const float3& a, const float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __host__ __device__ float Dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float Length(const float3& Vector)
{
    return sqrtf(Dot(Vector, Vector));
}

inline __host__ __device__ float3 Normalize(const float3& Vector)
{
    float Len = Length(Vector);
    return (Vector / Len);
}

inline __host__ __device__ float3 Cross(const float3& a, const float3& b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float3 make_float3(const float a)
{
    return make_float3(a, a, a);
}

inline __host__ __device__ float3 make_float3(const uint3& a)
{
    return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ bool operator==(const float3& a, const float3& b)
{
    const float Tolerance = 1e-3;
//    float3 tmpA = abs(a);
//    float3 tmpB = b;
//    if (tmpA.x <= Tolerance) { tmpA.x += 1.f; tmpB.x += 1.f; }
//    if (tmpA.y <= Tolerance) { tmpA.y += 1.f; tmpB.y += 1.f; }
//    if (tmpA.z <= Tolerance) { tmpA.z += 1.f; tmpB.z += 1.f; }

    const float3 tmp = abs((a - b));
    return (tmp.x < Tolerance && tmp.y < Tolerance && tmp.z < Tolerance);
}

/////////////////////////
/// float4
/////////////////////////

inline __host__ __device__ float4 make_float4(const float a)
{
    return make_float4(a, a, a, a);
}

#endif // CUDAMATH_H
