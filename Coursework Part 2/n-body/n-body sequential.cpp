#include <iostream>
#include <valarray>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <errno.h>


// Small value to avoid dividing by zero in force calculation
#define SOFTENING 1e-4f
// Gravitational constant
#define GRAV_CONST 6.67408e-11f

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
	float mass;
	MutableBody() {}
	MutableBody(const MVect3 &np, const MVect3 &nv, float m) {
		p = np;
		v = nv;
		mass = m;
	}
};

// ** FAILED ATTEMPT AT CREATING BMP IMAGE FROM POSITION DATA **
//struct lwrite
//{
//	unsigned long value;
//	unsigned size;
//
//	lwrite(unsigned long value, unsigned size) noexcept
//		: value(value), size(size)
//	{
//	}
//};
//
//inline std::ostream &operator<<(std::ostream &outs, const lwrite &v)
//{
//	unsigned long value = v.value;
//	for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
//		outs.put(static_cast<char>(value & 0xFF));
//	return outs;
//}
//
//bool array2bmp(const std::string &filename, const vector<MVect3> &pixels, const size_t width, const size_t height)
//{
//	std::ofstream f(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
//	if (!f)
//	{
//		return false;
//	}
//	// Write Bmp file headers
//	const size_t headers_size = 14 + 40;
//	const size_t padding_size = (4 - ((height * 3) % 4)) % 4;
//	const size_t pixel_data_size = width * ((height * 3) + padding_size);
//	f.put('B').put('M'); // bfType
//						 // bfSize
//	f << lwrite(headers_size + pixel_data_size, 4);
//	// bfReserved1, bfReserved2
//	f << lwrite(0, 2) << lwrite(0, 2);
//	// bfOffBits, biSize
//	f << lwrite(headers_size, 4) << lwrite(40, 4);
//	// biWidth,  biHeight,  biPlanes
//	f << lwrite(width, 4) << lwrite(height, 4) << lwrite(1, 2);
//	// biBitCount, biCompression = BI_RGB ,biSizeImage
//	f << lwrite(24, 2) << lwrite(0, 4) << lwrite(pixel_data_size, 4);
//	// biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant
//	f << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4);
//	// Write image data
//	for (size_t x = height; x > 0; x--)
//	{
//		for (size_t y = 0; y < width; y++)
//		{
//			const auto &val = pixels[((x - 1) * width) + y];
//			f.put(static_cast<char>(int(255.0 * val.z))).put(static_cast<char>(int(255.0 * val.y))).put(static_cast<char>(255.0 * val.x));
//		}
//		if (padding_size)
//		{
//			f << lwrite(0, padding_size);
//		}
//	}
//	return f.good();
//}
// ** FAILED ATTEMPT AT CREATING BMP IMAGE FROM POSITION DATA **

// ** FAILED PGM METHOD **
//void writePGM(const char *filename, const PGMData *data)
//{
//	FILE *pgmFile;
//	int i, j;
//	int hi, lo;
//
//	pgmFile = fopen(filename, "wb");
//	if (pgmFile == NULL) {
//		perror("cannot open file to write");
//		exit(EXIT_FAILURE);
//	}
//
//	fprintf(pgmFile, "P5 ");
//	fprintf(pgmFile, "%d %d ", data>col, data->row);
//	fprintf(pgmFile, "%d ", data->max_gray);
//
//	if (data->max_gray > 255) {
//		for (i = 0; i < data->row; ++i) {
//			for (j = 0; j < data->col; ++j) {
//				hi = HI(data->matrix[i][j]);
//				lo = LO(data->matrix[i][j]);
//				fputc(hi, pgmFile);
//				fputc(lo, pgmFile);
//			}
//
//		}
//	}
//	else {
//		for (i = 0; i < data->row; ++i) {
//			for (j = 0; j < data->col; ++j) {
//				lo = LO(data->matrix[i][j]);
//				fputc(lo, pgmFile);
//			}
//		}
//	}
//
//	fclose(pgmFile);
//	deallocate_dynamic_matrix(data->matrix, data->row);
//}
// **

// Body class - with initialise method and simulation method
class NBodyMutableClass {
private:
	int numBodies;
	float dt;
	valarray<MutableBody> bodies;
	valarray<MVect3> accel;

	void initBodies() {
		bodies.resize(numBodies);
		accel.resize(numBodies);
		bodies[0].p = MVect3(0, 0, 0);
		bodies[0].v = MVect3(0, 0, 0);
		bodies[0].mass = 1.0;
		for (int i = 1; i < numBodies; ++i) {
			bodies[i].p.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
			bodies[i].p.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
			bodies[i].p.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
			bodies[i].v.x = 0.0;
			bodies[i].v.y = 0.0;
			bodies[i].v.z = 0.0;
            bodies[i].mass = 10.0f * (rand() / (float)RAND_MAX);
		}
	}

public:
	NBodyMutableClass(int nb, float step) {
		numBodies = nb;
		dt = step;
		initBodies();
	}

	void forSim(int steps) {

		FILE* rdata;
		errno_t err;
		if ((err = fopen_s(&rdata, "rdata.py", "w")) != 0) {
			fprintf(stderr, "Couldn't open render.json");
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
					float dist = sqrt(dx*dx + dy*dy + dz*dz);
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
				results << i << "," << p.p.x << "," << p.p.y << "," << p.p.z << ", ," << p.v.x << "," << p.v.y << "," << p.v.z << "," << endl;
				//results << "Final accel: ," << accel[i].x << "," << accel[i].y << "," << accel[i].z << endl;
				p.v.x += accel[i].x*dt;
				p.v.y += accel[i].y*dt;
				p.v.z += accel[i].z*dt;
				p.p.x += p.v.x*dt;
				p.p.y += p.v.y*dt;
				p.p.z += p.v.z*dt;
                int x = ((p.p.x * 0.5f) + 0.5) * 800;
                int y = ((p.p.y * 0.5f) + 0.5) * 800;
                int r = 4;
				fprintf(rdata, "[%d, %d, %d],", x, y, r);
				
				fprintf(stderr, "Finished iterations %d.\n", i);
			}
			fprintf(rdata, "],\n");
			//array2bmp("img.bmp", positions, 1000, 1000);
		}
		fprintf(rdata, "]");
		fclose(rdata);
	}
};

int main(int argc, char *argv[]) {
	using namespace std::chrono;

	NBodyMutableClass sim(50, 0.001f);

	results << "Body, Pos x, Pos y, Pos z, , Vel x, Vel y, Vel z" << endl;
	//results << "Accel[i] x, y, z, , Accel[j] x, y, z" << endl;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	sim.forSim(1000);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<float> time_span = duration_cast<duration<float>>(t2 - t1);

	std::cout << "It took me " << time_span.count() << " seconds.\n";
	results << "\nTotal Time: ," << time_span.count() << endl;

	return 0;
}