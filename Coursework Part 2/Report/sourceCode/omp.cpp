void calcForces(Body *p, int numBodies) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i)
    {
        /* nested forloop force calculations */
    }