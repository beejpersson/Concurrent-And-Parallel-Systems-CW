for (size_t y = 0; y < size; ++y)
{
	for (size_t x = 0; x < size; ++x)
	{
		for (size_t sy = 0, i = (size - y - 1) * size + x; sy < 2; ++sy)
		{
			for (size_t sx = 0; sx < 2; ++sx)
			{
				for (size_t s = 0; s < samples; ++s)
				{ ... }	
			}
		}
	}
}