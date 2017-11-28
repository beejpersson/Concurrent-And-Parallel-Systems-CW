#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <errno.h>

// Small value to avoid dividing by zero in force calculation
#define SOFTENING 1e-4f
// Gravitational constant
#define GRAV_CONST 6.67408e-11f
// Pi
# define M_PI 3.14159265358979323846
// Block size
#define BLOCK_SIZE 256

// Used namespaces
using namespace std;
using namespace std::chrono;

// View cuda info
__global__ void cuda_info()
{
    // Get cuda device;
    int device;
    cudaGetDevice(&device);

    // Get device properties
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    //Display Properties
    cout << "Name: " << properties.name << endl;
    cout << "CUDA Capability: " << properties.major << endl;
    cout << "Cores: " << properties.multiProcessorCount << endl;
    cout << "Memory: " << properties.totalGlobalMem / (1024 * 1024) << "MB" << endl;
    cout << "Clock freq: " << properties.clockRate / 1000 << "MHz" << endl;
}

// vec3 struct used for body's position and velocity
struct MVect3 {
    float x, y, z;
    MVect3() {}
    MVect3(float nx, float ny, float nz) {
        x = nx;
        y = ny;
        z = nz;
    }
    void zero() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }
};

// Body struct - with position, velocity, mass and constructor
struct MutableBody {
    MVect3 p = { 0.0f,0.0f,0.0f }, v = { 0.0f,0.0f,0.0f };
    int mass;
    MutableBody() {}
    MutableBody(const MVect3 &np, const MVect3 &nv, int m) {
        p = np;
        v = nv;
        mass = m;
    }
};

// Body class - with initialise method and simulation method
class NBodyMutableClass {
private:
    int numBodies;
    float dt;
    vector<MutableBody> bodies;
    vector<MVect3> accel;

    // Set initial values of the bodies: positions: a random number between -1 and 1, velocities: 0, mass: between 0 and 100.
    void initBodies() {
        bodies.resize(numBodies);
        accel.resize(numBodies);
        bodies[0].p = MVect3(0, 0, 0);
        bodies[0].v = MVect3(0, 0, 0);
        bodies[0].mass = 1;
        for (int i = 1; i < numBodies; ++i) {
            bodies[i].p.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            bodies[i].p.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            bodies[i].p.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            bodies[i].v.x = 0.0;
            bodies[i].v.y = 0.0;
            bodies[i].v.z = 0.0;
            bodies[i].mass = rand() % 100 + 1;
        }
    }

public:
    NBodyMutableClass(int nb, float step) {
        numBodies = nb;
        dt = step;
        initBodies();
    }

    vector<MutableBody> get_bodies() {
        return bodies;
    }

    vector<MVect3> get_accels() {
        return accel;
    }

    void calcForces() {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numBodies) {
            accel[i].zero();
        }
        if (i < numBodies) {
            MutableBody & pi = bodies[i];
            for (int j = i + 1; j < numBodies; ++j) {
                MutableBody & pj = bodies[j];
                float dx = pi.p.x - pj.p.x;
                float dy = pi.p.y - pj.p.y;
                float dz = pi.p.z - pj.p.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                float magi = pj.mass / (dist*dist*dist + SOFTENING);
                accel[i].x -= magi*dx;
                accel[i].y -= magi*dy;
                accel[i].z -= magi*dz;
                float magj = pi.mass / (dist*dist*dist + SOFTENING);
                accel[j].x += magj*dx;
                accel[j].y += magj*dy;
                accel[j].z += magj*dz;

                //results << i << "," << accel[i].x << "," << accel[i].y << "," << accel[i].z << ", ," << j << "," << accel[j].x << "," << accel[j].y << "," << accel[j].z << endl;

                /*double dx = pj.p.x - pi.p.x;
                double dy = pj.p.y - pi.p.y;
                double dz = pj.p.z - pi.p.z;

                double distSqr = dx*dx + dy*dy + dz*dz;
                double invDist = 1.0f / (sqrt(distSqr)+SOFTENING);
                double invDist3 = invDist * invDist * invDist;

                accel[i].x += dx * invDist3; accel[i].y += dy * invDist3; accel[i].z += dz * invDist3;*/
            }
        }
    }

    void addForces()
    {
        for (int i = 0; i < numBodies; ++i) {
            MutableBody & p = bodies[i];
            // Before positions updated, print to results file
            //results << i << "," << p.p.x << "," << p.p.y << "," << p.p.z << ", ," << p.v.x << "," << p.v.y << "," << p.v.z << "," << endl;
            //results << "Final accel: ," << accel[i].x << "," << accel[i].y << "," << accel[i].z << endl;
            p.v.x += accel[i].x*dt;
            p.v.y += accel[i].y*dt;
            p.v.z += accel[i].z*dt;
            p.p.x += p.v.x*dt;
            p.p.y += p.v.y*dt;
            p.p.z += p.v.z*dt;
        }
    }
};

int main(int argc, char *argv[]) {
    // Initialise random seed
    srand(time(NULL));

    // Create results file
    ofstream results("data.csv", ofstream::out);

    // *** These parameters can be manipulated in the algorithm to modify work undertaken ***
    int numBodies = 100; // number of bodies
    int nIters = 100; // simulation iterations
    float timeStep = 0.0002f; // time step

    // * CUDA mem alloc *
    float *d_buf;
    cudaMalloc(&d_buf, numBodies);

    // Output headers to results file
    results << "Test, Number of Bodies, Simulation Iterations, Time, " << endl;

    int nBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run test iterations
    for (int i = 0; i < 10; ++i) {

        // Create json data file for visualisation
        FILE* rdata;
        errno_t err;
        if ((err = fopen_s(&rdata, "rdata.py", "w")) != 0) {
            fprintf(stderr, "Couldn't open rdata.py");
        }
        fprintf(rdata, "data = [\n");

        // Generate nbody sim, (num. bodies, timestep)
        NBodyMutableClass sim(*d_buf, timeStep);

        // * TIME FROM HERE... *
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (int step = 0; step < nIters; ++step) {

            cudaMemcpy(d_buf, &numBodies, (sizeof(float) * numBodies), cudaMemcpyHostToDevice);
            sim.calcForces();
            // Calculate forces applied to the bodies by each other
            cudaMemcpy(&numBodies, d_buf, (sizeof(float) * numBodies), cudaMemcpyHostToDevice);
            // Apply those forces and update the bodies positions
            sim.addForces();

            cuda_info<<<nBlocks, threads>>>();
            cudaDeviceSynchronize();
            //// ** Print positions of all bodies each step, for simulation renderer **
            //fprintf(rdata, "\t[");
            //for (int j = 0; j < sim.get_bodies().size(); ++j) {
            //    // Convert body positions to an int proportional to screen size to be sent to data file
            //    int x = ((sim.get_bodies()[j].p.x * 0.5f) + 0.5f) * 800.0f;
            //    int y = ((sim.get_bodies()[j].p.y * 0.5f) + 0.5f) * 800.0f;
            //    // Calculate radius from mass (assuming flat and equal densities of 1) to be send to data file
            //    int r = sqrt(sim.get_bodies()[j].mass / M_PI);
            //    fprintf(rdata, "[%d, %d, %d],", x, y, r);
            //}

            ////fprintf(stderr, "Finished iterations %d.\n", step);
            //fprintf(rdata, "],\n");
            //// ** Print positions of all bodies each step, for simulation renderer **

        }
        // * ...TO HERE *
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto time_span = duration_cast<milliseconds>(t2 - t1).count();

        fprintf(rdata, "]");
        fclose(rdata);

        // Output test no., variables and total time to results file
        results << i + 1 << ", " << numBodies << ", " << nIters << ", " << time_span << endl;

        // Output test iteration information to consolde outside of the timings to not slow algorithm
        cout << "Test " << i + 1 << " complete. Time = " << time_span << " milliseconds." << endl;
    }
    return 0;
}