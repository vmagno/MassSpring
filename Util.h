#ifndef UTIL_H__
#define UTIL_H__

#include <glm/glm.hpp>

#include "Common.h"
#include "CudaMath.h"

//template<typename Type> struct Vec2
//{
//    Type x, y;

//    Vec2(const Type Val = 0)
//        : x(Val), y(Val)
//    {}

//    Vec2(const Type NewX, const Type NewY)
//        : x(NewX), y(NewY)
//    {}

//    Vec2(const Vec2& Val)
//        : x(Val.x), y(Val.y)
//    {}

//};

//template<typename Type> struct Vec3
//{
//    Type x, y, z;

//    Vec3(const Type Val = 0)
//        : x(Val), y(Val), z(Val)
//    {}

//    Vec3(const Type NewX, const Type NewY, const Type NewZ)
//        : x(NewX), y(NewY), z(NewZ)
//    {}

//    Vec3(const Vec3& Val)
//        : x(Val.x), y(Val.y), z(Val.z)
//    {}


//    Type Dot(const Vec3& Other) const
//    {
//        return (x * Other.x) + (y * Other.y) + (z * Other.z);
//    }

//    Vec3 operator*(Type Scalar) const
//    {
//        return Vec3(x * Scalar, y * Scalar, z * Scalar);
//    }

//    Vec3 operator-(Vec3 Other) const
//    {
//        return Vec3(x - Other.x, y - Other.y, z - Other.z);
//    }
//};

//template<typename Type> struct Vec4
//{
//    Type x, y, z, w;

//    Vec4(const Type Val = 0)
//        : x(Val), y(Val), z(Val), w(Val)
//    {}

//    Vec4(const Type NewX, const Type NewY, const Type NewZ, const Type NewW)
//        : x(NewX), y(NewY), z(NewZ), w(NewW)
//    {}

//    Vec4(const Vec4& Val)
//        : x(Val.x), y(Val.y), z(Val.z), w(Val.w)
//    {}

//};

//typedef Vec2<uint> uint2;
//typedef Vec3<uint> uint3;
//typedef Vec3<float> float3;
//typedef Vec4<float> float4;

/**
 * @brief DecomposeVector Splits a vector into its projection on another vector and its perpendicular component
 * @param Vector The vector to split
 * @param Axis The axis onto which to project. THIS FUNCTION ASSUMES IT'S ALREADY NORMALIZED.
 * @param[out] AlongAxis The "parallel" component
 * @param[out] Perpendicular The "perpendicular" component
 */
inline void DecomposeVector(const float3& Vector, const float3& Axis, float3& AlongAxis, float3& Perpendicular)
{
    AlongAxis = Axis * Dot(Vector, Axis);
    Perpendicular = Vector - AlongAxis;
}

/**
 * @brief DecomposeVector Splits a vector into its projection on another vector and its perpendicular component
 * @param Vector The vector to split
 * @param Axis The axis onto which to project. THIS FUNCTION ASSUMES IT'S ALREADY NORMALIZED.
 * @param[out] AlongAxis The "parallel" component
 * @param[out] Perpendicular The "perpendicular" component
 */
inline void DecomposeVector(const glm::vec3& Vector, const glm::vec3& Axis, glm::vec3& AlongAxis, glm::vec3& Perpendicular)
{
    AlongAxis = Axis * (glm::dot(Vector, Axis));
    Perpendicular = Vector - Axis;
}

#endif // UTIL_H__
