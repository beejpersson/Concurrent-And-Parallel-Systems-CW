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

using namespace std;

// Create results file
ofstream results("data.csv", ofstream::out);

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

	// Set initial values of the bodies: positions: a random number between -1 and 1, velocities: 0, mass: between 0 and 1000.
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

	void forSim(int steps) {

        // Create json data file
		FILE* rdata;
		errno_t err;
		if ((err = fopen_s(&rdata, "rdata.py", "w")) != 0) {
			fprintf(stderr, "Couldn't open rdata.py");
		}
		fprintf(rdata, "data = [\n");

		for (int step = 0; step < steps; ++step) {
			for (int i = 0; i < numBodies; ++i) {
				accel[i].zero();
			}
			for (int i = 0; i < numBodies; ++i) {
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
			fprintf(rdata, "\t[");
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
                // Convert body positions to an int proportional to screen size to be sent to data file
				int x = ((p.p.x * 0.5f) + 0.5f) * 1280.0f;
				int y = ((p.p.y * 0.5f) + 0.5f) * 720.0f;
				// Calculate radius from mass (assuming equal densities of 1) to be send to data file
				int r = sqrt(p.mass/M_PI);
				fprintf(rdata, "[%d, %d, %d],", x, y, r);
				
				
				fprintf(stderr, "Finished iterations %d.\n", i);
			}
			fprintf(rdata, "],\n");
		}
		fprintf(rdata, "]");
		fclose(rdata);
	}
};

int main(int argc, char *argv[]) {
	using namespace std::chrono;

	NBodyMutableClass sim(50, 0.0002f);

	results << "Body, Pos x, Pos y, Pos z, , Vel x, Vel y, Vel z" << endl;
	//results << "Accel[i] x, y, z, , Accel[j] x, y, z" << endl;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	sim.forSim(1000);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<float> time_span = duration_cast<duration<float>>(t2 - t1);

	std::cout << "It took me " << time_span.count() << " seconds.\n";
	results << "\nTotal Time: ," << time_span.count() << endl;
	results << "\nBody's x Position: ," << sim.get_bodies()[49].p.x << endl;

	return 0;
}