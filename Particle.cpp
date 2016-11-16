#include "Particle.h"

using namespace glm;

Particle::Particle()
    : Id(-1),
      GridPos(make_uint3(0)),
      Position(0.f),
      Velocity(0.f),
      Force(0.f),
      Color(1.f),
      Mass(1.f),
      bFixed(false)
{
}

void Particle::ClampVelocity(float MaxSpeed)
{
    float Speed = length(Velocity);
    if (Speed > MaxSpeed)
    {
        Velocity = Velocity * MaxSpeed / Speed;
    }
}
