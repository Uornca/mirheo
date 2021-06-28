// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>
#include <mirheo/core/utils/path.h>
#include <cmath>
#include <random>
#include <chrono>

namespace mirheo
{

namespace integration_kernels
{

/**
 * \code transform(Particle& p, const real3 f, const real invm, const real dt) \endcode
 *  is a callable that performs integration. It is called for
 *  every particle and should change velocity and coordinate
 *  of the Particle according to the chosen integration scheme.
 *
 * Will read positions from \c oldPositions channel and write to positions
 * Will read velocities from velocities and write to velocities
 */
template<typename Transform>
__global__ void integrate(PVviewWithOldParticles pvView, const real dt, int edge, real localSize, Transform transform)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= pvView.size) return;

    real4 pos = readNoCache(pvView.oldPositions + pid);
    if (pos.x == 200 || pos.x == -200) {		// skip particle if it already exited left/right
    	writeNoCache(pvView.positions + pid, pos);
    	return;
    }
    real4 vel = readNoCache(pvView.velocities   + pid);
    Real3_int frc(pvView.forces[pid]);

    Particle p(pos, vel);

    transform(p, frc.v, pvView.invMass, dt);

    real4 tmp = p.r2Real4();
    real4 tmpv = p.u2Real4();
    if ((edge == 0 || edge == 4) && tmp.x <= -(localSize / 2)) { 	// check if particle exited left
	       tmp.x = -100.0;
          // tmpv.x = 0;
           //tmpv.y = 0;
           //tmpv.z = 0;
    } else if ((edge == 2 || edge == 4) && tmp.x >= (localSize / 2)) {  // check if particle exited right
	       tmp.x = 100.0;
           //tmpv.x = 0;
           //tmpv.y = 0;
           //tmpv.z = 0;
    } else {
        writeNoCache(pvView.velocities + pid, tmpv);

    }
	//writeNoCache(pvView.velocities + pid, tmpv);
    writeNoCache(pvView.positions  + pid, tmp);
}

template<typename Transform>
__global__ void calcMomentumOut(PVviewWithOldParticles viewNew, int edge, double bufferSize, real localSize, double *momentumOutLeft, double *momentumOutRight, int *leftBufferSize, int *rightBufferSize, int *gone, int *readyToReplace, Transform transform)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int leftB = 0;
    int rightB = 0;
    int departed = 0;
    real3 vel_r3, myMomentumLeft, myMomentumRight;
    vel_r3 = myMomentumLeft = myMomentumRight = make_real3(0._r);
    readyToReplace[pid] = 0;

    if (pid < viewNew.size) {
    	real tmp = readNoCache(viewNew.positions + pid).x;
    	if (tmp < 100 && tmp > -100) {
    	    if ((edge == 0 || edge == 4) && tmp < -(localSize/2)+bufferSize) {
    		leftB++;
    	    } else if ((edge == 2 || edge == 4) && tmp > (localSize/2)-bufferSize) {
    		rightB++;
    	    }
    	} else if (tmp == 100 || tmp == -100) {
    	    real4 pos = readNoCache(viewNew.positions + pid);
    	    real4 vel = readNoCache(viewNew.velocities + pid);
    	    vel_r3 = make_real3(vel);
    	    (tmp == 100) ? (myMomentumRight = vel_r3 * viewNew.mass) : (myMomentumLeft = vel_r3 * viewNew.mass);
    	    pos.x = pos.x * 2;
    	    writeNoCache(viewNew.positions + pid, pos);
            readyToReplace[pid] = 1;
    	}
    	if (tmp == -200 || tmp == 200) {
            departed++;
            readyToReplace[pid] = 1;
        }
    }

    myMomentumLeft = warpReduce(myMomentumLeft, [](real a, real b) { return a+b; });
    myMomentumRight = warpReduce(myMomentumRight, [](real a, real b) { return a+b; });
    leftB = warpReduce(leftB, [](int a, int b) { return a+b; });
    rightB = warpReduce(rightB, [](int a, int b) { return a+b; });
    departed = warpReduce(departed, [](int a, int b) { return a+b; });

    if (laneId() == 0) {
    	atomicAdd(momentumOutLeft+0, (double)myMomentumLeft.x);
    	atomicAdd(momentumOutLeft+1, (double)myMomentumLeft.y);
    	atomicAdd(momentumOutLeft+2, (double)myMomentumLeft.z);
    	atomicAdd(momentumOutRight+0, (double)myMomentumRight.x);
    	atomicAdd(momentumOutRight+1, (double)myMomentumRight.y);
    	atomicAdd(momentumOutRight+2, (double)myMomentumRight.z);
    	atomicAdd(leftBufferSize, (int)leftB);
    	atomicAdd(rightBufferSize, (int)rightB);
    	atomicAdd(gone, (int)departed);
    }

}

template<typename Transform>
__global__ void calcPotential(PVview pvView, real3 local, double *totalPotential, double *totalForce, Transform transform)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    real rc = 1.0;
    real invrc = 1 / rc;
    real myPotential = 0;
    real a = 10.3960396;
    real3 myForce = make_real3(0._r);
    int myPid = 0;
    if (pid < pvView.size) {
        real3 pos = make_real3(readNoCache(pvView.positions + pid));
        if (pos.x == 200) {
            myPid = pid;
        } else {
            real3 dr = pos - local;
            real rij2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
            if (rij2 < rc*rc) {
                real invrij = math::rsqrt(rij2);
                real rij = rij2 * invrij;
                real force = a * (1 - rij * invrc);
                //myPotential = a * rc * (1 - rij * invrc) * (1 - rij * invrc) / 2;
                myPotential = force * rc * (1 - rij * invrc) / 2;
                real forceFactor = force / rij;
                myForce = dr * forceFactor;
            }
        }
    }

    myPotential = warpReduce(myPotential, [](real a, real b) { return a+b; });
    myForce = warpReduce(myForce, [](real a, real b) { return a+b; });
    if (laneId() == 0) {
        atomicAdd(totalPotential, (real)myPotential);
        atomicAdd(totalForce+0, (real)myForce.x);
        atomicAdd(totalForce+1, (real)myForce.y);
        atomicAdd(totalForce+2, (real)myForce.z);
    }

}

template<typename Transform>
__global__ void actuallyInsert(PVview pvView, real3 local, real3 velocity, int replaceMe, double* momentumIn, Transform transform)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    real3 myMomentumIn = make_real3(0._r);
    if (pid == replaceMe) {
        real4 local4 = make_real4(local.x, local.y, local.z, 0);
        //local4.x = local.x;
        //local4.y = local.y;
        //local4.z = local.z;
        real4 vel4 = make_real4(velocity.x, velocity.y, velocity.z, 0);
        real4 pos = readNoCache(pvView.positions + pid);
        myMomentumIn = velocity * pvView.mass;
        momentumIn[0] = myMomentumIn.x;
        momentumIn[1] = myMomentumIn.y;
        momentumIn[2] = myMomentumIn.z;
        //printf("POSITION %f %f %f %f\n", pos.x, pos.y, pos.z, pos.w);
        //printf("NEWPOS %f %f %f %f\n", local4.x, local4.y, local4.z, local4.w);
        writeNoCache(pvView.positions  + pid, local4);
        writeNoCache(pvView.velocities  + pid, vel4);
        //printf("MOMIN %f %f %f\n", myMomentumIn.x, myMomentumIn.y, myMomentumIn.z);
    }
    /*myMomentumIn = warpReduce(myMomentumIn, [](real a, real b) { return a+b; });
    //printf("MOMIN %f %f %f\n", myMomentumIn.x, myMomentumIn.y, myMomentumIn.z);
    if (laneId() == 0) {
        atomicAdd(momentumIn+0, (double)myMomentumIn.x);
        atomicAdd(momentumIn+1, (double)myMomentumIn.y);
        atomicAdd(momentumIn+2, (double)myMomentumIn.z);
    }*/
}

template<typename Transform>
__global__ void applyExternalForce(PVview pvView, real dt, int edge, real localSize, real bufferSize, real momDiffLX, real momDiffLY, real momDiffLZ, real momDiffRX, real momDiffRY, real momDiffRZ, double p_external, double surface, int numLeft, int numRight, Transform transform)
{

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    real4 pos = readNoCache(pvView.positions + pid);

    if ((edge == 0 || edge == 4) && pos.x < -(localSize/2)+bufferSize) {
        int direction = 1;
        real3 momDiff;
        momDiff.x = momDiffLX;
        momDiff.y = momDiffLY;
        momDiff.z = momDiffLZ;
        real3 force3 = momDiff / dt + surface * p_external * direction;
        force3 /= numLeft;
        //if (pid < 5 && force3.x != 0) printf("FORCE %f %f %f POS %f\n", force3.x, force3.y, force3.z, pos.x);

        real4 force;
        force.x = force3.x;
        force.y = force3.y;
        force.z = force3.z;
        force.w = 0;
        real4 currentForce = readNoCache(pvView.forces + pid);
        //if (pid < 5 && force.x != 0) printf("before %f %f %f     ", currentForce.x, currentForce.y, currentForce.z);
        currentForce += force;
        //if (pid < 5 && force.x != 0) printf("after %f %f %f\n", currentForce.x, currentForce.y, currentForce.z);
    }

    if ((edge == 2 || edge == 4) && pos.x > (localSize/2)-bufferSize) {
        int direction = -1;
        real3 momDiff;
        momDiff.x = momDiffRX;
        momDiff.y = momDiffRY;
        momDiff.z = momDiffRZ;
        real3 force3 = momDiff / dt + surface * p_external * direction;
        force3 /= numRight;
        //if (pid < 5 && force3.x != 0) printf("FORCE %f %f %f POS %f\n", force3.x, force3.y, force3.z, pos.x);

        real4 force;
        force.x = force3.x;
        force.y = force3.y;
        force.z = force3.z;
        force.w = 0;
        real4 currentForce = readNoCache(pvView.forces + pid);
        //if (pid < 5 && force.x != 0) printf("before %f %f %f     ", currentForce.x, currentForce.y, currentForce.z);
        currentForce += force;
        //if (pid < 5 && force.x != 0) printf("after %f %f %f\n", currentForce.x, currentForce.y, currentForce.z);
    }

}
} // namespace integration_kernels

template<typename Transform>
static int insertParticles(PVview pvView, ParticleVector *pv, cudaStream_t stream, MPI_Comm comm, int myrank, int side, real deltaBuffer, int replaceMe, double* momIn, Transform transform)
{

    auto domain = pv->getState()->domain;
    double bufferSize = domain.bufferSize;

    int Nmap = 4;
    real uTarget = 1.93920 * 2 * Nmap;
    real absu0 = abs(uTarget);
    real uOverlap = pow(10, 4);
    real rsigma = 1.5;
    int itrials_max = 40;
    real displ = 1.0;
    real v_min = -2.83;
    real v_max = 2.83;
    real v_range = v_max - v_min;

    real vt;
    real vtold;
    real errv;
    //printf("RANK %d SIDE %d INSERT %d   %f\n", myrank, side, nInsert, deltaBuffer);
    //printf("MEMEME %d\n", replaceMe);

    // For num of particles
    //for (int iDelta = 0; iDelta < deltaBuffer; ++iDelta) {

    real3 globalCoords;
    real3 velocity;

	if (myrank == 0) {
        std::uniform_real_distribution<double> distVel(0, v_range);
        std::uniform_real_distribution<double> disty(0, domain.globalSize.y);
        std::uniform_real_distribution<double> distz(0, domain.globalSize.z);
        std::default_random_engine re (std::chrono::system_clock::now().time_since_epoch().count());
        if (side == 0) {
            std::uniform_real_distribution<double> distx(0, bufferSize);
            globalCoords.x = distx(re);
        } else {
            std::uniform_real_distribution<double> distx(domain.globalSize.x - bufferSize, domain.globalSize.x);
            globalCoords.x = distx(re);
        }
        globalCoords.y = disty(re);
        globalCoords.z = distz(re);
        velocity.x = 0; //distVel(re) + v_min;   // ALI 0
        velocity.y = 0; //distVel(re) + v_min;   // ALI 0
        velocity.z = 0; //distVel(re) + v_min;   // ALI 0
	}

	MPI_Bcast(&globalCoords, 3, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&velocity, 3, MPI_DOUBLE, 0, comm);

    real3 localCoords = domain.global2local(globalCoords);
    for (int i = 0; i < itrials_max; i++) {
        //printf("RANK: %d   GLOBAL: %f %f %f  LOCAL: %f %f %f\n", myrank, insertCoords[0], insertCoords[1], insertCoords[2], localCoords.x, localCoords.y, localCoords.z);

        PinnedBuffer<double> totalPotential{1};
        PinnedBuffer<double> totalForce{3};
        totalPotential.clear(stream);
        totalForce.clear(stream);
        SAFE_KERNEL_LAUNCH(
            integration_kernels::calcPotential,
            getNblocks(pvView.size, 128), 128, 0, stream,
            pvView, localCoords, totalPotential.devPtr(), totalForce.devPtr(), transform );

        totalPotential.downloadFromDevice(stream, ContainersSynch::Synch);
        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, totalPotential.data(), 1, MPI_DOUBLE, MPI_SUM, comm) );
        totalForce.downloadFromDevice(stream, ContainersSynch::Synch);
        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, totalForce.data(), 3, MPI_DOUBLE, MPI_SUM, comm) );
        //if (myrank == 0) printf("POTENTIAL: %f    FORCE: %f %f %f\n",
        //                        totalPotential[0], totalForce[0], totalForce[1], totalForce[2]);

        vt = totalPotential[0];
        if (vt == 0 || vt < uTarget + pow(10, -3) ) {
            //printf("SUCCESS %d\n", i);
            //INSERT PARTICLE
            if(side == myrank) {
                PinnedBuffer<double> momentumIn{3};
                momentumIn.clear(stream);
                SAFE_KERNEL_LAUNCH(
                    integration_kernels::actuallyInsert,
                    getNblocks(pvView.size, 128), 128, 0, stream,
                    pvView, localCoords, velocity, replaceMe, momentumIn.devPtr(), transform );

                momentumIn.downloadFromDevice(stream, ContainersSynch::Synch);
                //printf("MOMENTUMIN %f %f %f\n", momentumIn[0], momentumIn[1], momentumIn[2]);
                momIn[0] = momentumIn[0];
                momIn[1] = momentumIn[1];
                momIn[2] = momentumIn[2];
                return 1;
            }
            break;
        }
    }
    return 0;
    //}

}

template<typename Transform>
static void integrate(ParticleVector *pv, real dt, Transform transform, cudaStream_t stream, MPI_Comm comm)
{
    constexpr int nthreads = 128;

    // New particles now become old
    std::swap(pv->local()->positions(), *pv->local()->dataPerParticle.getData<real4>(channel_names::oldPositions));
    PVviewWithOldParticles pvView(pv, pv->local());

    auto domain = pv->getState()->domain;
    int numPart = pv->local()->positions().size();
    double bufferSize = domain.bufferSize;
    real localSize = domain.localSize.x;
    int edge;
    if (domain.globalStart.x == 0 && domain.globalStart.x + localSize == domain.globalSize.x) {
	       edge = 4;
    } else if (domain.globalStart.x <= bufferSize) {
	       edge = 0;
    } else if (domain.globalStart.x + localSize >= domain.globalSize.x - bufferSize) {
	       edge = 2;
    } else {
	       edge = 1;
    }

    SAFE_KERNEL_LAUNCH(
        integration_kernels::integrate,
        getNblocks(pvView.size, nthreads), nthreads, 0, stream,
        pvView, dt, edge, localSize, transform );

    PinnedBuffer<double> momentumOutLeft{3};
    PinnedBuffer<double> momentumOutRight{3};
    PinnedBuffer<int> leftBufferSize{1};
    PinnedBuffer<int> rightBufferSize{1};
    PinnedBuffer<int> gone{1};
    PVviewWithOldParticles viewNew(pv, pv->local());
    PinnedBuffer<int> readyToReplace{numPart};
    momentumOutLeft.clear(stream);
    momentumOutRight.clear(stream);
    leftBufferSize.clear(stream);
    rightBufferSize.clear(stream);
    gone.clear(stream);
    readyToReplace.clear(stream);

    SAFE_KERNEL_LAUNCH(
    	integration_kernels::calcMomentumOut,
    	getNblocks(viewNew.size, nthreads), nthreads, 0, stream,
    	viewNew, edge, bufferSize, localSize, momentumOutLeft.devPtr(), momentumOutRight.devPtr(), leftBufferSize.devPtr(), rightBufferSize.devPtr(), gone.devPtr(), readyToReplace.devPtr(), transform );

    momentumOutLeft.downloadFromDevice(stream, ContainersSynch::Synch);
    momentumOutRight.downloadFromDevice(stream, ContainersSynch::Synch);
    leftBufferSize.downloadFromDevice(stream, ContainersSynch::Synch);
    rightBufferSize.downloadFromDevice(stream, ContainersSynch::Synch);
    gone.downloadFromDevice(stream, ContainersSynch::Synch);
    readyToReplace.downloadFromDevice(stream, ContainersSynch::Synch);

    int myrank;
    int mpiSize;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &mpiSize);
    MPI_Check( MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : momentumOutLeft.data(), momentumOutLeft.data(), 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : momentumOutRight.data(), momentumOutRight.data(), 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, leftBufferSize.data(), 1, MPI_INT, MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, rightBufferSize.data(), 1, MPI_INT, MPI_SUM, comm) );
    MPI_Check( MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : gone.data(), gone.data(), 1, MPI_INT, MPI_SUM, 0, comm) );
    //MPI_Check( MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : readyToReplace.data(), readyToReplace.data(), viewNew.size, MPI_INT, MPI_SUM, 0, comm) );

    int replaceable = 0;
    for (int i = 0; i < numPart; i++) {
        if (readyToReplace[i] == 1)
        //printf("HERE %d : %d\n", i+(myrank*numPart), readyToReplace[i]);
        replaceable++;
    }
    //printf("REPLACE %d\n", replaceable);

    // Check if need insert
    real bufferAlpha = 0.95;
    real bufferTau = 0.005;
    int bufferAverage = 1350;
    real deltaBufferLeft = (bufferAlpha * bufferAverage - leftBufferSize[0]) * dt / bufferTau;
    real deltaBufferRight = (bufferAlpha * bufferAverage - rightBufferSize[0]) * dt / bufferTau;
    /*if (myrank == 0) {
    	printf("LEFT BUFFER: %d   RIGHT BUFFER: %d  MOMENTUM LEFT: %g %g %g  MOMENTUM RIGHT: %g %g %g  GONE: %d STEP: %lld DELTA_L: %g DELTA_R: %g\n",
    	leftBufferSize[0], rightBufferSize[0], momentumOutLeft[0], momentumOutLeft[1], momentumOutLeft[2], momentumOutRight[0], momentumOutRight[1], momentumOutRight[2], gone[0], pv->getState()->currentStep, deltaBufferLeft, deltaBufferRight);
    }*/
    double momentumInLeft[3] = { 0, 0, 0 };
    double momentumInRight[3] = { 0, 0, 0 };
    if (deltaBufferLeft > 0) {
        int replaceMe;
        for (int i = 0; i < numPart; i++) {
            if (readyToReplace[i] == 1) {
                replaceMe = i;
                break;
            }
        }
        leftBufferSize[0] +=insertParticles(viewNew, pv, stream, comm, myrank, 0, deltaBufferLeft, replaceMe, momentumInLeft, transform);
        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, leftBufferSize.data(), 1, MPI_INT, MPI_MAX, comm) );
        //printf("MOMLEFT %f %f %f\n", momentumInLeft[0], momentumInLeft[1], momentumInLeft[2]);
    }
    if (deltaBufferRight > 0) {
        int replaceMe;
        for (int i = 0; i < numPart; i++) {
            if (readyToReplace[i] == 1) {
                replaceMe = i;
                break;
            }
        }
        rightBufferSize[0] += insertParticles(viewNew, pv, stream, comm, myrank, 1, deltaBufferRight, replaceMe, momentumInRight, transform);
        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, rightBufferSize.data(), 1, MPI_INT, MPI_MAX, comm) );
        //printf("MOMRIGHT %f %f %f\n", momentumInRight[0], momentumInRight[1], momentumInRight[2]);
    }

    // RAZLIKA GIBALNIH KOLICIN PO VSTAVLJANJU
    // ZUNANJA SILA GLEDE NA TO RAZLIKO
    real3 momDiffLeft;
    momDiffLeft.x = momentumOutLeft[0] - momentumInLeft[0];
    momDiffLeft.y = momentumOutLeft[1] - momentumInLeft[1];
    momDiffLeft.z = momentumOutLeft[2] - momentumInLeft[2];
    //printf("MOMDIFFLEFT %f %f %f\n", momDiffLeft.x, momDiffLeft.y, momDiffLeft.z);

    real3 momDiffRight;
    momDiffRight.x = momentumOutRight[0] - momentumInRight[0];
    momDiffRight.y = momentumOutRight[1] - momentumInRight[1];
    momDiffRight.z = momentumOutRight[2] - momentumInRight[2];
    //printf("MOMDIFFRIGHT %f %f %f\n", momDiffRight.x, momDiffRight.y, momDiffRight.z);

    /*
    enotski_x = smer (proti noter)
    A = y * z od celice
    P_external = konstanta (hardcode)
    */

    if (momDiffLeft.x != 0 || momDiffLeft.y != 0 || momDiffLeft.z != 0 || momDiffRight.x != 0 || momDiffRight.y != 0 || momDiffRight.z != 0) {
        int surface = domain.globalSize.y * domain.globalSize.z;
        double p_external = 188.8;

        SAFE_KERNEL_LAUNCH(
            integration_kernels::applyExternalForce,
            getNblocks(viewNew.size, 128), 128, 0, stream,
            viewNew, dt, edge, localSize, bufferSize, momDiffLeft.x, momDiffLeft.y, momDiffLeft.z, momDiffRight.x, momDiffRight.y, momDiffRight.z, p_external, surface, leftBufferSize[0], rightBufferSize[0], transform );
    }
    // __global__ void applyExternalForce(PVview pvView, real dt, real3 momentumDiff, double p_external, double surface, int numParts, Transform transform)


    /*if (myrank == 0) {
    	printf("LEFT BUFFER: %d   RIGHT BUFFER: %d  MOMENTUM LEFT: %g %g %g  MOMENTUM RIGHT: %g %g %g  GONE: %d STEP: %lld DELTA_L: %g DELTA_R: %g\n",
    	leftBufferSize[0], rightBufferSize[0], momentumOutLeft[0], momentumOutLeft[1], momentumOutLeft[2], momentumOutRight[0], momentumOutRight[1], momentumOutRight[2], gone[0], pv->getState()->currentStep, deltaBufferLeft, deltaBufferRight);
    }*/
}

} // namespace mirheo
