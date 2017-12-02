#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
#include <fstream>

// Small value to avoid dividing by zero in force calculation
#define SOFTENING 1e-4f
// Gravitational constant
#define GRAV_CONST 6.67408e-11f
// Pi
# define M_PI 3.14159265358979323846

// Used namespaces
using namespace std;
using namespace std::chrono;

// vec3 struct used for body's position and velocity
struct Body { float x, y, z, vx, vy, vz, ax, ay, az, mass; };

// Method to initialise values of a body, random position between -1 and 1, 0 velocity and accel, random mass betwen 1 and 100
void initBodies(Body *p, int n) {
    for (int i = 0; i < n; ++i) {
        Body & pi = p[i];
        pi.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pi.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pi.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pi.vx = 0.0f; pi.vy = 0.0f; pi.vz = 0.0f;
        pi.ax = 0.0f; pi.ay = 0.0f; pi.az = 0.0f;
        pi.mass = rand() % 100 + 1;
    }
}

// Method to calculate the forces acting on each body based on their distances from each other
void calcForces(Body *p, int n) {
    for (int i = 0; i < n; ++i) {
        Body & pi = p[i];
        for (int j = i + 1; j < n; ++j) {
            Body & pj = p[j];
            float dx = pi.x - pj.x;
            float dy = pi.y - pj.y;
            float dz = pi.z - pj.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float magi = pj.mass / (dist*dist*dist + SOFTENING);
            pi.ax -= magi*dx;
            pi.ay -= magi*dy;
            pi.az -= magi*dz;
            float magj = pi.mass / (dist*dist*dist + SOFTENING);
            pj.ax += magj*dx;
            pj.ay += magj*dy;
            pj.az += magj*dz;
        }
    }
}

int main(int argc, char *argv[]) {
    // Initialise random seed
    srand(time(NULL));

    // Create results file
    ofstream results("data.csv", ofstream::out);

    // *** These parameters can be manipulated in the algorithm to modify work undertaken ***
    int numBodies = 300; // number of bodies
    int nIters = 1000; // simulation iterations
    float timeStep = 0.0002f; // time step

    // Output headers to results file
    results << "Test, Number of Bodies, Simulation Iterations, Time, " << endl;

    // Run test iterations
    for (int i = 0; i < 1; ++i) {

        // Create json data file for visualisation
        FILE* rdata;
        errno_t err;
        if ((err = fopen_s(&rdata, "rdata.py", "w")) != 0) {
            fprintf(stderr, "Couldn't open rdata.py");
        }
        fprintf(rdata, "data = [\n");

        // Allocate memory for all bodies, (num. bodies, timestep)
        int bytes = numBodies * sizeof(Body);
        float *buf = (float*)malloc(bytes);
        Body *p = (Body*)buf;

        // Initialise values for bodies
        initBodies(p, numBodies);

        // * TIME FROM HERE... *
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (int step = 0; step < nIters; ++step) {

            // Calculate forces applied to the bodies by each other
            calcForces(p, numBodies);

            // Update positions based on calculated forces and reset acceleration
            for (int j = 0; j < numBodies; ++j) {
                Body & pj = p[j];
                pj.vx += pj.ax*timeStep;
                pj.vy += pj.ay*timeStep;
                pj.vz += pj.az*timeStep;
                pj.x += pj.vx*timeStep;
                pj.y += pj.vy*timeStep;
                pj.z += pj.vz*timeStep;
                pj.ax = 0.0f; pj.ay = 0.0f; pj.az = 0.0f;
            }

            // ** Print positions of all bodies each step, for simulation renderer **
            fprintf(rdata, "\t[");
            for (int j = 0; j < numBodies; ++j) {
                Body & pj = p[j];
                // Convert body positions to an int proportional to screen size to be sent to data file
                int x = ((pj.x * 0.5f) + 0.5f) * 800.0f;
                int y = ((pj.y * 0.5f) + 0.5f) * 800.0f;
                // Calculate radius from mass (assuming flat and equal densities of 1) to be send to data file
                int r = sqrt(pj.mass / M_PI);
                fprintf(rdata, "[%d, %d, %d],", x, y, r);
            }

            //fprintf(stderr, "Finished iterations %d.\n", step);
            fprintf(rdata, "],\n");
            // ** Print positions of all bodies each step, for simulation renderer **

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