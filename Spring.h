#ifndef SPRING_H
#define SPRING_H

#include "Util.h"

class Particle;

class Spring
{
public:
    Spring();

    void SetParticles(Particle* New1, Particle* New2);
    void SetRestLength(float NewLength) { RestLength = NewLength; }
    void SetCoefficient(float NewCoef) { Coefficient = NewCoef; }

    void GetParticles(Particle*& First, Particle*& Second) { First = p1; Second = p2; }
    float GetRestLength() const { return RestLength; }
    float GetCoefficient() const { return Coefficient; }

    uint2 GetIds() { return make_uint2(p1->GetId(), p2->GetId()); }

private:

    Particle* p1;
    Particle* p2;

    float RestLength;
    float Coefficient;
};

#endif // SPRING_H
