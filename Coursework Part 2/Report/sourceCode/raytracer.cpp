// Required for manual threading to allow forLoopAlgorithm method to update pixel vector
vector<vec> pixels;

// Nested for loop method, to be manually multi-threaded
void forLoopAlgorithm(unsigned int threads, size_t dimension, unsigned int num_threads, size_t samples, _Binder<_Unforced, uniform_real_distribution<double>&, default_random_engine&> get_random_number, vec r, vec cx, vec cy, vector<sphere> spheres, ray camera)
{
	for (size_t y = threads; y < dimension; y += num_threads)
	{
		for (size_t x = 0; x < dimension; ++x)
		{
			for (size_t sy = 0, i = (dimension - y - 1) * dimension + x; sy < 2; ++sy)
			{
				for (size_t sx = 0; sx < 2; ++sx)
				{
					r = vec();
					for (size_t s = 0; s < samples; ++s)
					{
						double r1 = 2 * get_random_number(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * get_random_number(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						vec direction = cx * static_cast<double>(((sx + 0.5 + dx) / 2 + x) / dimension - 0.5) + cy * static_cast<double>(((sy + 0.5 + dy) / 2 + y) / dimension - 0.5) + camera.direction;
						r = r + radiance(spheres, ray(camera.origin + direction * 140, direction.normal()), 0) * (1.0 / samples);
					}
					pixels[i] = pixels[i] + vec(clamp(r.x, 0.0, 1.0), clamp(r.y, 0.0, 1.0), clamp(r.z, 0.0, 1.0)) * 0.25;
				}
			}
		}
	}
}

int main(int argc, char **argv)
{
	random_device rd;
	default_random_engine generator(rd());
	uniform_real_distribution<double> distribution;
	auto get_random_number = bind(distribution, generator);

	// *** These parameters can be manipulated in the algorithm to modify work undertaken ***
	constexpr size_t dimension = 256;
	constexpr size_t samples = 16; // Algorithm performs 4 * samples per pixel.
	vector<sphere> spheres
	{
		sphere(1e5, vec(1e5 + 1, 40.8, 81.6), vec(), vec(0.75, 0.25, 0.25), reflection_type::DIFFUSE),
		sphere(1e5, vec(-1e5 + 99, 40.8, 81.6), vec(), vec(0.25, 0.25, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, 1e5), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, -1e5 + 170), vec(), vec(), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 1e5, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, -1e5 + 81.6, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(16.5, vec(27, 16.5, 47), vec(), vec(1, 1, 1) * 0.999, reflection_type::SPECULAR),
		sphere(16.5, vec(73, 16.5, 78), vec(), vec(1, 1, 1) * 0.999, reflection_type::REFRACTIVE),
		sphere(600, vec(50, 681.6 - 0.27, 81.6), vec(12, 12, 12), vec(), reflection_type::DIFFUSE)
	};
	// **************************************************************************************

	// Create results file
	ofstream results("data.csv", ofstream::out);

    // Output headers to results file
	results << "Test, Image Dimensions, Samples Per Pixel, Time, " << endl;

    // Run test iterations
	for (unsigned int j = 0; j < 100; ++j)
	{
		ray camera(vec(50, 52, 295.6), vec(0, -0.042612, -1).normal());
		vec cx = vec(0.5135);
		vec cy = (cx.cross(camera.direction)).normal() * 0.5135;
		vec r;

        // Required for manual threading
		pixels.resize(dimension * dimension);

        // Required for OpenMP
		//vector<vec> pixels(dimension * dimension);
		//int y;

		// * TIME FROM HERE... *
		auto start = system_clock::now();

		// *** MANUAL MULTITHREADING ***
		vector<thread> threads;
		
		for (unsigned int t = 0; t < 4; ++t)
		{
			threads.push_back(thread(forLoopAlgorithm, t, dimension, 4, samples, get_random_number, r, cx, cy, spheres, camera));
		}
		for (auto &t : threads)
			t.join();

//		// *** OPENMP *** (change scheduling to "dynamic" or "static")
//#pragma omp parallel for num_threads(4) private(y, r) schedule(dynamic)
//		for (y = 0; y < dimension; ++y)
//		{
//			for (size_t x = 0; x < dimension; ++x)
//			{
//				for (size_t sy = 0, i = (dimension - y - 1) * dimension + x; sy < 2; ++sy)
//				{
//					for (size_t sx = 0; sx < 2; ++sx)
//					{
//						r = vec();
//						for (size_t s = 0; s < samples; ++s)
//						{
//							double r1 = 2 * get_random_number(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
//							double r2 = 2 * get_random_number(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
//							vec direction = cx * static_cast<double>(((sx + 0.5 + dx) / 2 + x) / dimension - 0.5) + cy * static_cast<double>(((sy + 0.5 + dy) / 2 + y) / dimension - 0.5) + camera.direction;
//							r = r + radiance(spheres, ray(camera.origin + direction * 140, direction.normal()), 0) * (1.0 / samples);
//						}
//						pixels[i] = pixels[i] + vec(clamp(r.x, 0.0, 1.0), clamp(r.y, 0.0, 1.0), clamp(r.z, 0.0, 1.0)) * 0.25;
//					}
//				}
//			}
//		}

		// * ...TO HERE *
		auto end = system_clock::now();
		auto total = duration_cast<milliseconds>(end - start).count();
        
        // Output test no., variables and total time to results file
		results << j + 1 << ", " << dimension << ", " << samples * 4 << ", " << total << endl;

        // Output test information to consolde outside of the timings to not slow algorithm
		cout << "Test " << j + 1 << " complete. Time = " << total << "." <<  num_threads << endl;
		array2bmp("img.bmp", pixels, dimension, dimension);

        // Required for manual threading
		pixels.clear();
	}

	return 0;
}