for (size_t y = 0; y < dimension; ++y)
{
	for (size_t x = 0; x < dimension; ++x)
	{
		for (size_t sy = 0, i = (dimension - y - 1) * dimension + x; sy < 2; ++sy)
		{
			for (size_t sx = 0; sx < 2; ++sx)
			{
				for (size_t s = 0; s < samples; ++s)
				{ ... }	
			}
		}
	}
}