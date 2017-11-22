#pragma omp parallel for num_threads(num_threads) private(y, r) schedule(dynamic)
    for (y = 0; y < dimension; ++y)
    {
        .../*nested for loops*/
    }