// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "minimize.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

IntegratorMinimize::IntegratorMinimize(const MirState *state, const std::string& name, real maxDisplacement) :
    Integrator(state, name), maxDisplacement_{maxDisplacement}
{}

IntegratorMinimize::IntegratorMinimize(const MirState *state, Loader&, const ConfigObject& object) :
    IntegratorMinimize(state, object["name"], object["maxDisplacement"])
{}

void IntegratorMinimize::execute(ParticleVector *pv, cudaStream_t stream, MPI_Comm comm)
{
    const auto t  = static_cast<real>(getState()->currentTime);
    const auto dt = static_cast<real>(getState()->getDt());

    auto st2 = [max = maxDisplacement_] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        // Limit the displacement magnitude to `max`.
        real3 dr = dt * dt * invm * f;
        real dr2 = dot(dr, dr);
        if (dr2 > max * max)
            dr *= max * math::rsqrt(dr2);
        p.r += dr;
    };

    integrate(pv, dt, st2, stream, comm);
    invalidatePV_(pv);
}

void IntegratorMinimize::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject(this, _saveSnapshot(saver, "IntegratorMinimize"));
}

ConfigObject IntegratorMinimize::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Integrator::_saveSnapshot(saver, typeName);
    config.emplace("maxDisplacement", saver(maxDisplacement_));
    return config;
}

} // namespace mirheo
