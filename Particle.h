#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>

#include "Util.h"

class Particle
{
public:
    Particle();

    void SetId(uint NewId) { Id = NewId; }
    void SetPosition(glm::vec3 NewPos) { Position = NewPos; }
    void SetVelocity(glm::vec3 NewVel) { Velocity = NewVel; }
    void SetForce(glm::vec3 NewForce) { Force = NewForce; }
    void SetColor(glm::vec4 NewCol) { Color = NewCol; }
    void SetColor(glm::vec3 NewCol, float NewAlpha) { Color = glm::vec4(NewCol, NewAlpha); }
    void SetGridPos(uint3 NewPos) { GridPos = NewPos; }
    void SetMass(float NewMass) { Mass = NewMass; }
    void SetFixed(bool bNewFixed = true) { bFixed = bNewFixed; }

    uint GetId() { return Id; }
    glm::vec3 GetPosition() { return Position; }
    glm::vec3 GetVelocity() { return Velocity; }
    glm::vec3 GetForce() { return Force; }
    glm::vec4 GetColor() { return Color; }
    uint3 GetGridPos() { return GridPos; }
    float GetMass() { return Mass; }
    bool IsFixed() { return bFixed; }

    float GetDistance(const Particle& Other) { return glm::length(Position - Other.Position); }
    glm::vec3 GetVector(const Particle& Other) { return (Position - Other.Position); }

    void ClampVelocity(float MaxSpeed);

private:

    uint Id;
    uint3 GridPos;
    glm::vec3 Position;
    glm::vec3 Velocity;
    glm::vec3 Force;
    glm::vec4 Color;
    float Mass;

    bool bFixed;

};

#endif // PARTICLE_H
