#ifndef MASSSPRINGSYSTEM_H
#define MASSSPRINGSYSTEM_H

#include <map>
#include <vector>

#include <GL/glew.h>

#include "BasicTimer.h"
#include "HostDeviceCode.h"
#include "Particle.h"
#include "Spring.h"


class MassSpringSystem
{
public:
    MassSpringSystem();
    ~MassSpringSystem();

    void AddParticle(Particle p);
    void AddSpring(Spring s);

    void GenerateCube(uint NumX, uint NumY, uint NumZ, glm::vec3 Dimension);
    void InitGL(GLint locVertex, GLint locColor);
    void InitCuda();

    void Draw();

    /**
     * @brief Advance the system by one timestep
     * @param DeltaT Size of the timestep in seconds
     */
    void UpdateSystem(float DeltaT);

    void SetFixedTop(bool bFixedTop = true) { Param.bFixedTop = bFixedTop; }
    void ToggleFixedTop() { Param.bFixedTop = !Param.bFixedTop; }
    void TogglePrintTime() { bPrintTimers = !bPrintTimers; }

private:

    std::vector<Particle> Particles; //!< The particles
    std::vector<Spring> Springs;

    uint3 GridSize;

    MSSParameters Param;

    GLuint Vao;
    uint NumVBO;
    GLuint* VBOs;
//    GLuint VboVertex;
//    GLuint VboColor;
//    GLuint VboSpring;
#ifdef DEBUG_VELOCITY
    GLuint VaoVel;
#endif

    GLuint LocVertex, LocColor;

    BasicTimer ForceComputationTime;
    BasicTimer IntegrationTime;
    BasicTimer BufferUpdateTime;

    bool bPrintTimers;

    float TotalMass;

    Arrays HostArrays;
    Arrays DeviceArrays;

    cudaGraphicsResource** VboCudaResources;

    /**
     * @brief Update the VBO with new particle positions
     */
    void UpdateParticleBuffer();

    /**
     * @brief Compute new particle positions and velocities based on current forces
     */
    void IntegrateSystem(float DeltaT);

    void ApplyForces();
    void AddSpringForces();

    void AllocArrays();
    void FillArrays();
    void FreeArrays();

    void** GetArrayAddress(uint ArrayId);
    void MapBuffers();
    void UnmapBuffers();


   // void PrintStats();
};

#endif // MASSSPRINGSYSTEM_H
