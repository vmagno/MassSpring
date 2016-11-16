#include "MassSpringSystem.h"

#include <iomanip>
#include <iostream>

#include <cuda_gl_interop.h>

#include "Kernels.h"

using namespace glm;
using namespace std;

enum {
    VERTEX_VBO_ID = 0,
    COLOR_VBO_ID,
    SPRING_VBO_ID,
    VEL_VBO_ID,
    VEL_COLOR_VBO_ID
};

MassSpringSystem::MassSpringSystem()
    :
      Vao(0),
      NumVBO(3),
      LocVertex(0),
      LocColor(0),
      bPrintTimers(false),
      TotalMass(0.f)
{
    Particles.clear();
    Springs.clear();

    HostArrays.ClearPtrs();
    DeviceArrays.ClearPtrs();

    Param.init();

#ifdef DEBUG_VELOCITY
    NumVBO = 5;
#endif

    VBOs = new GLuint[NumVBO];
    VboCudaResources = new cudaGraphicsResource*[NumVBO];

    // Sanity check
    if (!is_pod<Arrays>::value)
    {
        cerr << "[WARNING] Arrays struct is not POD (see HostDeviceCode.h)" << endl;
    }

    if (!is_pod<MSSParameters>::value)
    {
        cerr << "[WARNING] MSSParameters struct is not POD (see HostDeviceCode.h)" << endl;
    }
}

void MassSpringSystem::AddParticle(Particle p)
{
    p.SetId(Particles.size());
    Particles.push_back(p);

    TotalMass += p.GetMass();
//    MassDistribution[ceilf(p.GetMass() * 1000)]++;
}

void MassSpringSystem::AddSpring(Spring s)
{
    Springs.push_back(s);

//    cout << "Adding spring with length " << s.GetRestLength() << " and strength " << s.GetCoefficient() << endl;

//    SpringDistribution[ceilf(s.GetCoefficient() * 1000)]++;
//    SpringLengthDistribution[ceilf(s.GetRestLength() * 1000)]++;
}

void MassSpringSystem::GenerateCube(uint NumX, uint NumY, uint NumZ, glm::vec3 Dimension)
{
    GridSize = make_uint3(NumX, NumY, NumZ);
    vec3 MinPos = -Dimension / 2.f;
    MinPos.z = 0.12f;

    if (NumX < 2 || NumY < 2 || NumZ < 2)
    {
        cerr << "[WARNING] Number of particles is too small along 1 axis" << endl;
    }

    const uint TotalNumBlocks = (NumX - 1) * (NumY - 1) * (NumZ - 1);
    const float ParticleBaseMass = Param.TotalMass / TotalNumBlocks / 8.f;
    const uint TotalNumParticles = NumX * NumY * NumZ;
    const float ParticleMass = Param.TotalMass / TotalNumParticles;

    vec3 NumSections(NumX, NumY, NumZ);
    if (NumX > 1) NumSections.x--;
    if (NumY > 1) NumSections.y--;
    if (NumZ > 1) NumSections.z--;
    vec3 Increment = Dimension / NumSections;
    vec3 ColorIncr = vec3(1.f) / vec3(NumX, NumY, NumZ);

    for (uint k = 0; k < NumZ; k++)
    {
        for (uint j = 0; j < NumY; j++)
        {
            for (uint i = 0; i < NumX; i++)
            {
                Particle p;
                p.SetPosition(MinPos + (Increment * vec3(i, j, k)));
                p.SetColor(ColorIncr * vec3(i+1, j+1, k+1), 1.f);
                p.SetGridPos(make_uint3(i, j, k));

                uint MassMultiplier = 1;
                if (i > 0 && i < NumX - 1) MassMultiplier *= 2;
                if (j > 0 && j < NumY - 1) MassMultiplier *= 2;
                if (k > 0 && k < NumZ - 1) MassMultiplier *= 2;
                p.SetMass(ParticleBaseMass * MassMultiplier);
//                p.SetMass(ParticleMass);

                p.SetColor(vec3((ParticleBaseMass*8 - p.GetMass())/(ParticleBaseMass*8), 0.f, p.GetMass()/(ParticleBaseMass*8)), 1.f);

                if (k == NumZ - 1 && Param.bFixedTop) p.SetFixed(true);

//                 if (i == 0 && (k == NumZ - 1)) p.SetFixed(true);
//                if (i == (NumX - 1) && (k == NumZ - 1)) p.SetFixed(true);

                AddParticle(p);
            }
        }
    }


    const float BaseLength = Dimension.x / NumX;
    if (Increment.x != Increment.y || Increment.x != Increment.z)
    {
        cerr << "[WARNING] Not using cubic units!" << endl;
    }

    //////////////////////////////
    // Stretch/compression springs
    {
        const float Coefficient =
                (Param.YoungModulus * BaseLength * (4 * Param.PoissonRatio + 1))
                / (8 * (Param.PoissonRatio + 1));


        for (uint i = 0; i < Particles.size(); i++)
        {
            const uint3 GridPos(Particles[i].GetGridPos());

            if (GridPos.x < NumX - 1)
            {
                int Multiplier = 1;
                if (GridPos.y > 0 && GridPos.y < NumY - 1) Multiplier *= 2;
                if (GridPos.z > 0 && GridPos.z < NumZ - 1) Multiplier *= 2;

                Spring s;
                s.SetParticles(&Particles[i], &Particles[i + 1]);
                //            s.SetCoefficient(Param.BaseSpringCoef * Multiplier);
                s.SetCoefficient(Coefficient * Multiplier);
                AddSpring(s);
            }

            if (GridPos.y < NumY - 1)
            {
                int Multiplier = 1;
                if (GridPos.x > 0 && GridPos.x < NumX - 1) Multiplier *= 2;
                if (GridPos.z > 0 && GridPos.z < NumZ - 1) Multiplier *= 2;
                Spring s;
                s.SetParticles(&Particles[i], &Particles[i + NumX]);
                s.SetCoefficient(Coefficient * Multiplier);
                AddSpring(s);
            }

            if (GridPos.z < NumZ - 1)
            {
                int Multiplier = 1;
                if (GridPos.x > 0 && GridPos.x < NumX - 1) Multiplier *= 2;
                if (GridPos.y > 0 && GridPos.y < NumY - 1) Multiplier *= 2;
                Spring s;
                s.SetParticles(&Particles[i], &Particles[i + (NumX * NumY)]);
                s.SetCoefficient(Coefficient * Multiplier);
                AddSpring(s);
            }
        }
    }

    /////////////////////
    // Shear springs
    {
        const float Coefficient =
                (3 * Param.YoungModulus * BaseLength)
                / (8 * (Param.PoissonRatio + 1));

        for (uint i = 0; i < Particles.size(); i++)
        {
            const uint3 GridPos(Particles[i].GetGridPos());

            if (GridPos.x < NumX -1 && GridPos.y < NumY - 1 && GridPos.z < NumZ - 1)
            {
                Spring s1, s2, s3, s4;
                s1.SetParticles(&Particles[i], &Particles[i + (NumX * (NumY+1)) + 1]);
                s2.SetParticles(&Particles[i+1],&Particles[i + (NumX * (NumY+1))]);
                s3.SetParticles(&Particles[i + NumX], &Particles[i + (NumX * NumY) + 1]);
                s4.SetParticles(&Particles[i + (NumX * NumY)], &Particles[i + NumX + 1]);

//                s1.SetCoefficient(Param.BaseSpringCoef * 10);
//                s2.SetCoefficient(Param.BaseSpringCoef * 10);
//                s3.SetCoefficient(Param.BaseSpringCoef * 10);
//                s4.SetCoefficient(Param.BaseSpringCoef * 10);
                s1.SetCoefficient(Coefficient);
                s2.SetCoefficient(Coefficient);
                s3.SetCoefficient(Coefficient);
                s4.SetCoefficient(Coefficient);

                AddSpring(s1);
                AddSpring(s2);
                AddSpring(s3);
                AddSpring(s4);
            }
        }
    }

    Param.NumParticles = Particles.size();
    Param.NumSprings = Springs.size();

    //PrintStats();

    LaunchTest(4);
}

void MassSpringSystem::InitGL(GLint locVertex, GLint locColor)
{
    LocVertex = locVertex;
    LocColor = locColor;

    glGenBuffers(NumVBO, VBOs);

    glGenVertexArrays(1, &Vao);
    glBindVertexArray(Vao);

    // Vertices
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[VERTEX_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, Particles.size() * sizeof(float3), 0, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(LocVertex, 3, GL_FLOAT, GL_FALSE, 0, NULL );
    glEnableVertexAttribArray(LocVertex);

    // Particle colors
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[COLOR_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, Particles.size() * sizeof(float4), 0, GL_STATIC_DRAW);
    glVertexAttribPointer( LocColor, 4, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray( LocColor );

    // Spring connectivity
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOs[SPRING_VBO_ID]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, Springs.size() * sizeof(uint2), 0, GL_STATIC_DRAW);

    glBindVertexArray(0);

#ifdef DEBUG_VELOCITY
    glGenVertexArrays(1, &VaoVel);
    glBindVertexArray(VaoVel);

    // Velocity vector positions
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[VEL_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, Particles.size() * sizeof(float3) * 2, 0, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(LocVertex, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(LocVertex);

    // Velocity vector colors
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[VEL_COLOR_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, Particles.size() * sizeof(float4) * 2, 0, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(LocColor, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(LocColor);

    glBindVertexArray(0);
#endif
}

void MassSpringSystem::UpdateParticleBuffer()
{
    for (uint i = 0; i < Particles.size(); i++)
    {
        const vec3 pos = Particles[i].GetPosition();
        HostArrays.Positions[i].x = pos.x;
        HostArrays.Positions[i].y = pos.y;
        HostArrays.Positions[i].z = pos.z;
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBOs[VERTEX_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, Particles.size() * sizeof(float3), HostArrays.Positions, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MassSpringSystem::Draw()
{
    glBindVertexArray(Vao);

    glPointSize(10.f);
    glDrawArrays(GL_POINTS, 0, Particles.size());
    glLineWidth(2.f);
    glDrawElements(GL_LINES, Springs.size() * 2, GL_UNSIGNED_INT, 0);
    glLineWidth(1.f);

    glBindVertexArray(0);

#ifdef DEBUG_VELOCITY
    glBindVertexArray(VaoVel);
    glDrawArrays(GL_LINES, 0, Particles.size() * 2);
    glBindVertexArray(0);
#endif
}

void MassSpringSystem::ApplyForces()
{
#ifdef CPU_COMPUTATIONS
    for (uint i = 0; i < Particles.size(); i++)
    {
        const vec3 pos = Particles[i].GetPosition();
        vec3 NewForce = vec3(0.f); //Particles[i].GetForce();

        // Gravity
        NewForce += vec3(0.f, 0.f, -Param.Gravity);

        // Floor
        if (pos.z < Param.FloorHeight)
        {
            NewForce += vec3(0.f, 0.f, (Param.FloorHeight - pos.z) * Param.FloorStrength);
        }

        Particles[i].SetForce(NewForce);
    }
#endif // CPU_COMPUTATIONS

    LaunchApplyForces(Param, DeviceArrays);

    AddSpringForces();
}

void MassSpringSystem::AddSpringForces()
{
#ifdef CPU_COMPUTATIONS
    for (uint i = 0; i < Springs.size(); i++)
    {
        Particle *p1, *p2;
        Springs[i].GetParticles(p1, p2);
        vec3 Vector = p1->GetVector(*p2);
        const float Distance = glm::length(Vector);
        const vec3 Direction = Vector / Distance;
        float DeltaLength = Springs[i].GetRestLength() - Distance;

#if 0
        const float CompressionRatio = DeltaLength / Springs[i].GetRestLength();
        if (CompressionRatio < -0.08f)
        {
            //cout << "DeltaLength went from " << DeltaLength;
            DeltaLength -= DeltaLength * (CompressionRatio + 0.08f) * 20.f;
            //cout << " to " << DeltaLength << endl;
        }
        else if (CompressionRatio > 0.08f)
        {
            cout << "DeltaLength went from " << DeltaLength;
            DeltaLength += DeltaLength * (CompressionRatio - 0.08f) * 20.f;
            cout << " to " << DeltaLength << endl;
        }
#endif

        const vec3 BaseForce = Direction * DeltaLength * Springs[i].GetCoefficient();

//        p1->SetForce((p1->GetForce() + BaseForce));
//        p2->SetForce((p2->GetForce() - BaseForce));
        p1->SetForce(p1->GetForce() + (BaseForce / p1->GetMass()));
        p2->SetForce(p2->GetForce() - (BaseForce / p2->GetMass()));

        // Remove velocity towards each other if the spring is too compressed (or too stretched)
//        if (Distance / Springs[i].GetRestLength() < 0.1f || Distance / Springs[i].GetRestLength() > 2.5f)
//        {
//            vec3 AlongAxis1, Perp1, AlongAxis2, Perp2;
//            DecomposeVector(p1->GetVelocity(), Direction, AlongAxis1, Perp1);
//            DecomposeVector(p2->GetVelocity(), Direction, AlongAxis2, Perp2);

//            vec3 Axial = AlongAxis1 + AlongAxis2;
//            p1->SetVelocity(Perp1 + Axial);
//            p2->SetVelocity(Perp2 + Axial);
//        }

    }

#endif // CPU_COMPUTATIONS

    LaunchAddSpringForces(Param, DeviceArrays);
}

void MassSpringSystem::AllocArrays()
{
    HostArrays.Positions = new float3[Particles.size()];
    HostArrays.ParticleColors = new float4[Particles.size()];
    HostArrays.ParticleMasses = new float[Particles.size()];
    HostArrays.SpringConnect = new uint2[Springs.size()];
    HostArrays.RestLengths = new float[Springs.size()];
    HostArrays.Coefficients = new float[Springs.size()];
    HostArrays.SpringColors = new float4[Springs.size() * 2];

    //CudaCheck(cudaMalloc(&DeviceArrays.Positions,      Particles.size() * sizeof(float3)));
    CudaCheck(cudaMalloc(&DeviceArrays.Velocities,     Particles.size() * sizeof(float3)));
    CudaCheck(cudaMalloc(&DeviceArrays.Forces,         Particles.size() * sizeof(float3)));
    CudaCheck(cudaMalloc(&DeviceArrays.ParticleMasses, Particles.size() * sizeof(float)));
    //CudaCheck(cudaMalloc(&DeviceArrays.ParticleColors, Particles.size() * sizeof(float4)));
    //CudaCheck(cudaMalloc(&DeviceArrays.SpringConnect,  Springs.size() * sizeof(uint2)));
    CudaCheck(cudaMalloc(&DeviceArrays.RestLengths,    Springs.size() * sizeof(float)));
    CudaCheck(cudaMalloc(&DeviceArrays.Coefficients,   Springs.size() * sizeof(float)));
    CudaCheck(cudaMalloc(&DeviceArrays.SpringColors,   Springs.size() * sizeof(float4) * 2));
}

void MassSpringSystem::FillArrays()
{
    for (uint i = 0; i < Particles.size(); i++)
    {
        vec3 pos = Particles[i].GetPosition();
        HostArrays.Positions[i].x = pos.x;
        HostArrays.Positions[i].y = pos.y;
        HostArrays.Positions[i].z = pos.z;

        vec4 col = Particles[i].GetColor();
        HostArrays.ParticleColors[i].x = col.x;
        HostArrays.ParticleColors[i].y = col.y;
        HostArrays.ParticleColors[i].z = col.z;
        HostArrays.ParticleColors[i].w = col.w;

        HostArrays.ParticleMasses[i] = Particles[i].GetMass();
    }

    for (uint i = 0; i < Springs.size(); i++)
    {
        HostArrays.SpringConnect[i] = Springs[i].GetIds();
        HostArrays.RestLengths[i] = Springs[i].GetRestLength();
        HostArrays.Coefficients[i] = Springs[i].GetCoefficient();
    }


    CudaCheck(cudaMemcpy(DeviceArrays.Positions, HostArrays.Positions, Particles.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemset(DeviceArrays.Velocities, 0, Particles.size() * sizeof(float3)));
    CudaCheck(cudaMemset(DeviceArrays.Forces, 0, Particles.size() * sizeof(float3)));
    CudaCheck(cudaMemcpy(DeviceArrays.ParticleMasses, HostArrays.ParticleMasses, Particles.size() * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(DeviceArrays.ParticleColors, HostArrays.ParticleColors, Particles.size() * sizeof(float4), cudaMemcpyHostToDevice));

    CudaCheck(cudaMemcpy(DeviceArrays.SpringConnect, HostArrays.SpringConnect, Springs.size() * sizeof(uint2), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(DeviceArrays.RestLengths, HostArrays.RestLengths, Springs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(DeviceArrays.Coefficients, HostArrays.Coefficients, Springs.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void MassSpringSystem::FreeArrays()
{
    if (HostArrays.Positions) { delete[] HostArrays.Positions; }
    if (HostArrays.Velocities) { delete[] HostArrays.Velocities; }
    if (HostArrays.Forces) { delete[] HostArrays.Forces; }
    if (HostArrays.ParticleMasses) { delete[] HostArrays.ParticleMasses; }
    if (HostArrays.ParticleColors) { delete[] HostArrays.ParticleColors; }
    if (HostArrays.SpringConnect) { delete[] HostArrays.SpringConnect; }
    if (HostArrays.RestLengths) { delete[] HostArrays.RestLengths; }
    if (HostArrays.Coefficients) { delete[] HostArrays.Coefficients; }
    if (HostArrays.SpringColors) { delete[] HostArrays.SpringColors; }

    HostArrays.ClearPtrs();

    DeviceArrays.Positions = NULL;
    DeviceArrays.ParticleColors = NULL;
    DeviceArrays.SpringConnect = NULL;
    DeviceArrays.FreeDeviceMem();
}

void** MassSpringSystem::GetArrayAddress(uint ArrayId)
{
    switch(ArrayId)
    {
    default:
        cerr << "Trying to use a non-existing array!" << endl;
        return NULL;
    case VERTEX_VBO_ID:
        return (void**)&DeviceArrays.Positions;
    case COLOR_VBO_ID:
        return (void**)&DeviceArrays.ParticleColors;
    case SPRING_VBO_ID:
        return (void**)&DeviceArrays.SpringConnect;
#ifdef DEBUG_VELOCITY
    case VEL_VBO_ID:
        return (void**)&DeviceArrays.VelocityVectors;
    case VEL_COLOR_VBO_ID:
        return (void**)&DeviceArrays.VelocityColors;
#endif
    }
}

void MassSpringSystem::MapBuffers()
{
    for (uint iBuffer = 0; iBuffer < NumVBO; iBuffer++)
    {
        size_t NumBytes;
        CudaCheck(cudaGraphicsMapResources(1, &VboCudaResources[iBuffer], 0));
        CudaCheck(cudaGraphicsResourceGetMappedPointer(GetArrayAddress(iBuffer), &NumBytes, VboCudaResources[iBuffer]));
    }
}

void MassSpringSystem::UnmapBuffers()
{
    for (uint iBuffer = 0; iBuffer < NumVBO; iBuffer++)
    {
        CudaCheck(cudaGraphicsUnmapResources(1, &VboCudaResources[iBuffer]));
    }
}

void MassSpringSystem::InitCuda()
{
    for (uint iBuffer = 0; iBuffer < NumVBO; iBuffer++)
    {
        CudaCheck(cudaGraphicsGLRegisterBuffer(&VboCudaResources[iBuffer], VBOs[iBuffer], cudaGraphicsMapFlagsNone));
    }

    MapBuffers();

    AllocArrays();
    FillArrays();

    UnmapBuffers();
}

void MassSpringSystem::IntegrateSystem(float DeltaT)
{
    Param.DeltaT = DeltaT;
#ifdef CPU_COMPUTATIONS
    for (uint i = 0; i < Particles.size(); i++)
    {
        Particles[i].SetVelocity(Particles[i].GetVelocity() + Particles[i].GetForce() * DeltaT);
        Particles[i].ClampVelocity(15.f);

        if (!Particles[i].IsFixed())
            Particles[i].SetPosition(Particles[i].GetVelocity() * DeltaT + Particles[i].GetPosition());

        Particles[i].SetVelocity(Particles[i].GetVelocity() * Param.Damping); // Damping
    }
#endif

    LaunchIntegrateSystem(Param, DeviceArrays);
}

void MassSpringSystem::UpdateSystem(float DeltaT)
{
    MapBuffers();

    CudaCheck(cudaThreadSynchronize());

    ForceComputationTime.Start();
    ApplyForces();
    ForceComputationTime.Stop();

#ifdef CPU_COMPUTATIONS
//    float3 VelTmp[Particles.size()];
//    CudaCheck(cudaMemcpy(VelTmp, DeviceArrays.Velocities, Particles.size() * sizeof(float3), cudaMemcpyDeviceToHost));
//    for(uint i = 0; i < Particles.size(); i++)
//    {
//        const vec3 tmp = Particles[i].GetVelocity();
//        float3 vel2 = make_float3(tmp.x, tmp.y, tmp.z);
//        if (!(VelTmp[i] == vel2))
//        {
//            cout << "Difference: " << VelTmp[i].x << ", " << VelTmp[i].y << ", " << VelTmp[i].z << " -- "
//                 << vel2.x << ", " << vel2.y << ", " << vel2.z
//                 << endl;
//        }
//    }

//    float3 ForceTmp[Particles.size()];
//    CudaCheck(cudaMemcpy(ForceTmp, DeviceArrays.Forces, Particles.size() * sizeof(float3), cudaMemcpyDeviceToHost));
//    for(uint i = 0; i < Particles.size(); i++)
//    {
//        const vec3 tmp = Particles[i].GetForce();
//        float3 for2 = make_float3(tmp.x, tmp.y, tmp.z);
//        if (!(ForceTmp[i] == for2))
//        {
//            cout << "Difference: " << ForceTmp[i].x << ", " << ForceTmp[i].y << ", " << ForceTmp[i].z << " -- "
//                 << for2.x << ", " << for2.y << ", " << for2.z
//                 << endl;
//        }
//    }
#endif

    IntegrationTime.Start();
    IntegrateSystem(DeltaT);
    IntegrationTime.Stop();

    BufferUpdateTime.Start();
    //UpdateParticleBuffer();
    BufferUpdateTime.Stop();

    if (bPrintTimers)
    {
        cout << "Timing info (ms): " << endl;
        cout << setprecision(3);
        cout << "DeltaT:        " <<  DeltaT * 1000 << endl;
        cout << "ApplyForces:   " << ForceComputationTime.GetAverageTimeMs() << endl;
        cout << "Integration:   " << IntegrationTime.GetAverageTimeMs() << endl;
        cout << "Buffer update: " << BufferUpdateTime.GetAverageTimeMs() << endl;
        bPrintTimers = false;
    }


    UnmapBuffers();
}


MassSpringSystem::~MassSpringSystem()
{
    Particles.clear();
    Springs.clear();

    CudaCheck(cudaThreadSynchronize());

    glDeleteBuffers(NumVBO, VBOs);
    delete[] VBOs;


    for (uint i = 0; i < NumVBO; i++)
    {
        CudaCheck(cudaGraphicsUnregisterResource(VboCudaResources[i]));
    }
    delete[] VboCudaResources;

    FreeArrays();

    CudaCheck(cudaDeviceReset());

}
