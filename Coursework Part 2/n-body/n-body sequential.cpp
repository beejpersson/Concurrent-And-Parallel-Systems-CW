#include <iostream>
#include <valarray>
#include <cmath>
#include <chrono>
#include <fstream>

using namespace std;

struct MVect3 {
	double x, y, z;
	MVect3() {}
	MVect3(double nx, double ny, double nz) {
		x = nx;
		y = ny;
		z = nz;
	}
	void zero() {
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}
};

struct MutableBody {
	MVect3 p = { 0,0,0 }, v = { 0,0,0 };
	double mass;
	MutableBody() {}
	MutableBody(const MVect3 &np, const MVect3 &nv, double m) {
		p = np;
		v = nv;
		mass = m;
	}
};

class NBodyMutableClass {
	int numBodies;
	double dt;
	valarray<MutableBody> bodies;
	valarray<MVect3> accel;

	void initBodies() {
		bodies.resize(numBodies);
		accel.resize(numBodies);
		bodies[0].p = MVect3(0, 0, 0);
		bodies[0].v = MVect3(0, 0, 0);
		bodies[0].mass = 1.0;
		for (int i = 1; i < numBodies; ++i) {
			bodies[i].p.x = i;
			bodies[i].p.y = 0.0;
			bodies[i].p.z = 0.0;
			bodies[i].v.x = 0.0;
			bodies[i].v.y = sqrt(1.0 / i);
			bodies[i].v.z = 0.0;
		}
	}

public:
	NBodyMutableClass(int nb, double step) {
		numBodies = nb;
		dt = step;
		initBodies();
	}

	void forSim(int steps) {
		for (int step = 0; step < steps; ++step) {
			for (int i = 0; i < numBodies; ++i) {
				accel[i].zero();
			}
			for (int i = 0; i < numBodies; ++i) {
				MutableBody & pi = bodies[i];
				for (int j = i + 1; j < numBodies; ++j) {
					MutableBody & pj = bodies[j];
					double dx = pi.p.x - pj.p.x;
					double dy = pi.p.y - pj.p.y;
					double dz = pi.p.z - pj.p.z;
					double dist = sqrt(dx*dx + dy*dy + dz*dz);
					double magi = pj.mass / (dist*dist*dist);
					accel[i].x -= magi*dx;
					accel[i].y -= magi*dy;
					accel[i].z -= magi*dz;
					double magj = pi.mass / (dist*dist*dist);
					accel[j].x += magj*dx;
					accel[j].y += magj*dy;
					accel[j].z += magj*dz;
				}
			}
			for (int i = 0; i < numBodies; ++i) {
				MutableBody & p = bodies[i];
				p.v.x += accel[i].x*dt;
				p.v.y += accel[i].y*dt;
				p.v.z += accel[i].z*dt;
				p.p.x += p.v.x*dt;
				p.p.y += p.v.y*dt;
				p.p.z += p.v.z*dt;
			}
		}
		//			cout << bodies[500].p.x << "\n";
	}
};

int main(int argc, char *argv[]) {
	using namespace std::chrono;

	// Create results file
	ofstream results("data.csv", ofstream::out);

	NBodyMutableClass sim(1000, 0.01);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	sim.forSim(1000);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

	std::cout << "It took me " << time_span.count() << " seconds.\n";
	results << "It took me " << time_span.count() << " seconds." << endl;


	return 0;
}