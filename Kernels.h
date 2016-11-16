#ifndef KERNELS_H
#define KERNELS_H

#include "Common.h"
#include "HostDeviceCode.h"

extern void LaunchTest(uint SomeValue);

extern void LaunchApplyForces(MSSParameters Param, Arrays DeviceArrays);
extern void LaunchAddSpringForces(MSSParameters Param, Arrays DeviceArrays);
extern void LaunchIntegrateSystem(MSSParameters Param, Arrays DeviceArrays);

#endif // KERNELS_H
