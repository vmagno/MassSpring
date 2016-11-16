#include "Particle.h"
#include "Spring.h"

Spring::Spring()
{
    p1 = nullptr;
    p2 = nullptr;

    RestLength = 1.f;
    Coefficient = 0.f;
}

void Spring::SetParticles(Particle* New1, Particle* New2)
{
    p1 = New1;
    p2 = New2;

    SetRestLength(p1->GetDistance(*p2));
}
